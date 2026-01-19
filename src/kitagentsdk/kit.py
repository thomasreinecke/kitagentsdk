# src/kitagentsdk/kit.py
import os
import requests
import json
import logging
import sys
import threading
import time
from queue import Queue
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

class KitClient:
    """A client for interacting with the Kit API from within an agent."""
    def __init__(self):
        self.api_endpoint = os.getenv("KIT_API_ENDPOINT")
        self.api_key = os.getenv("KIT_API_KEY")
        self.run_id = os.getenv("KIT_RUN_ID") # Can be None for local runs
        self.agent = None # This will be set by BaseAgent
        
        if not all([self.api_endpoint, self.api_key]):
            # Use stderr for local runs as logger might not be configured by the agent script yet
            print("--- [SDK-WARN] KitClient is missing KIT_API_ENDPOINT or KIT_API_KEY. API calls will be disabled. Check .env file. ---", file=sys.stderr)
            self.enabled = False
        else:
            self.enabled = True
            self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
            if not self.run_id:
                print("--- [SDK] KitClient initialized for local run (no KIT_RUN_ID detected). ---", file=sys.stderr)
            else:
                # Start telemetry background thread
                self._metrics_queue = Queue()
                self._log_queue = Queue()
                self._progress_queue = Queue() 
                self._stop_event = threading.Event()
                self._telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
                self._telemetry_thread.start()

    def _telemetry_worker(self):
        """Background worker to flush metrics, logs, and progress."""
        while not self._stop_event.is_set():
            self._flush_metrics()
            self._flush_logs()
            self._flush_progress()
            time.sleep(1.0) 
        
        self._flush_metrics()
        self._flush_logs()
        self._flush_progress()

    def _flush_metrics(self):
        if self._metrics_queue.empty(): return
        batch = []
        while not self._metrics_queue.empty() and len(batch) < 100:
            batch.append(self._metrics_queue.get())
        if not batch: return
        payload = {"run_id": self.run_id, "metrics": batch}
        try:
            requests.post(f"{self.api_endpoint}/api/telemetry/metrics", json=payload, headers=self.headers, timeout=5)
        except Exception as e:
            print(f"[SDK-ERR] Failed to push metrics: {e}", file=sys.stderr)

    def _flush_logs(self):
        while not self._log_queue.empty():
            msg = self._log_queue.get()
            payload = {"run_id": self.run_id, "message": msg}
            try:
                requests.post(f"{self.api_endpoint}/api/telemetry/log", json=payload, headers=self.headers, timeout=5)
            except Exception as e:
                print(f"[SDK-ERR] Failed to push log: {e}", file=sys.stderr)
    
    def _flush_progress(self):
        latest_step = None
        while not self._progress_queue.empty():
            latest_step = self._progress_queue.get()
        if latest_step is not None:
            payload = {"run_id": self.run_id, "step": latest_step}
            try:
                requests.post(f"{self.api_endpoint}/api/telemetry/progress", json=payload, headers=self.headers, timeout=5)
            except Exception as e:
                print(f"[SDK-ERR] Failed to push progress: {e}", file=sys.stderr)

    def shutdown(self):
        """Stops the telemetry worker."""
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
            if self._telemetry_thread.is_alive():
                self._telemetry_thread.join(timeout=2.0)

    # --- Telemetry Methods ---

    def log_message(self, message: str):
        """Queues a log message."""
        if self.enabled and self.run_id:
            self._log_queue.put(message)
        else:
            print(message, flush=True)

    def log_metric(self, name: str, step: int, value: float):
        """Queues a metric."""
        if self.enabled and self.run_id:
            self._metrics_queue.put({"step": step, "name": name, "value": value})
            
    def log_progress(self, step: int):
        """Queues a progress update."""
        if self.enabled and self.run_id:
            self._progress_queue.put(step)
        
    def log_event(self, event_name: str, status: str = "info"):
        """Sends a lifecycle event immediately."""
        if self.enabled and self.run_id:
            payload = {"event": event_name, "status": status}
            try:
                requests.post(f"{self.api_endpoint}/api/telemetry/event/{self.run_id}", json=payload, headers=self.headers, timeout=5)
            except Exception as e:
                print(f"[SDK-ERR] Failed to send event {event_name}: {e}", file=sys.stderr)
        else:
            print(f"[EVENT] {event_name} ({status})", flush=True)

    # --- Data & Configuration ---

    def get_run_config(self) -> dict | None:
        """Fetches the configuration for the current run ID from the API."""
        if not self.enabled or not self.run_id:
            return None
        
        endpoint = f"{self.api_endpoint}/api/runs/detail"
        try:
            response = requests.post(endpoint, json={"id": self.run_id}, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("config", {})
        except Exception as e:
            print(f"[SDK-ERR] Failed to fetch run config: {e}", file=sys.stderr)
            return None

    def upload_artifact(self, file_path: str, artifact_type: str = 'generic'):
        if not self.enabled or not self.run_id:
            print(f"[SDK] Cannot upload artifact {file_path}: SDK disabled.", file=sys.stderr)
            return

        endpoint = f"{self.api_endpoint}/api/runs/{self.run_id}/artifacts"
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                data = {'artifact_type': artifact_type}
                response = requests.post(endpoint, files=files, data=data, headers=self.headers, timeout=600)
                response.raise_for_status()
        except Exception as e:
            print(f"[SDK-ERR] Failed to upload artifact {file_path}: {e}", file=sys.stderr)

    def download_artifact(self, artifact_id: UUID, destination_path: str | Path) -> bool:
        if not self.enabled: return False
        endpoint = f"{self.api_endpoint}/api/artifacts/{artifact_id}/download"
        try:
            if self.agent: self.agent.emit_event("ARTIFACT_DOWNLOAD_STARTED")
            with requests.get(endpoint, headers={"X-API-KEY": self.api_key}, stream=True) as r:
                r.raise_for_status()
                with open(destination_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            if self.agent: self.agent.emit_event("ARTIFACT_DOWNLOAD_COMPLETED", "success")
            return True
        except Exception as e:
            msg = f"Failed to download artifact: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            return False
            
    def download_artifacts_for_run(self, source_run_id: UUID, destination_folder: Path) -> bool:
        list_endpoint = f"{self.api_endpoint}/api/runs/{source_run_id}/artifacts/list"
        try:
            list_response = requests.post(list_endpoint, headers=self.headers, timeout=10)
            list_response.raise_for_status()
            artifacts = list_response.json()
        except Exception as e:
            msg = f"Failed to list artifacts: {e}"
            if self.agent: self.agent.log(msg)
            return False

        if not artifacts: return False

        for artifact in artifacts:
            self.download_artifact(artifact['id'], destination_folder / artifact['filename'])
        return True

    def get_training_data(self, params: dict) -> dict | None:
        # 1. Local Cache Check
        local_data_path = os.getenv("KIT_LOCAL_DATA_PATH")
        if local_data_path and os.path.exists(local_data_path):
            msg = f"--- [SDK] Using locally injected data from {local_data_path} ---"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            try:
                with open(local_data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed local load: {e}", file=sys.stderr)

        # 2. API Fallback
        if not self.enabled: return None
        
        endpoint = f"{self.api_endpoint}/api/data/training_set"
        try:
            if self.agent: self.agent.emit_event("TRAINING_DATA_REQUESTED")
            response = requests.post(endpoint, json=params, headers=self.headers, timeout=300)
            response.raise_for_status()
            if self.agent: self.agent.emit_event("TRAINING_DATA_RECEIVED", "success")
            return response.json()
        except Exception as e:
            msg = f"Failed to get training data: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            if self.agent: self.agent.emit_event("TRAINING_DATA_FAILED", "failure")
            return None