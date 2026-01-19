# src/kitagentsdk/kit.py
import os
import requests
import json
import logging
import sys
import threading
import time
from queue import Queue, Empty
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

class KitClient:
    """A client for interacting with the Kit API from within an agent."""
    def __init__(self):
        self.api_endpoint = os.getenv("KIT_API_ENDPOINT")
        self.api_key = os.getenv("KIT_API_KEY")
        self.run_id = os.getenv("KIT_RUN_ID")
        self.agent = None 
        
        # Strip trailing slash if present to avoid double slashes in URLs
        if self.api_endpoint and self.api_endpoint.endswith('/'):
            self.api_endpoint = self.api_endpoint[:-1]
        
        if not all([self.api_endpoint, self.api_key]):
            print("--- [SDK-WARN] KitClient missing credentials. API disabled. ---", file=sys.stderr)
            self.enabled = False
        else:
            self.enabled = True
            # Default headers for JSON endpoints
            self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
            
            if not self.run_id:
                print("--- [SDK] Initialized in Local Mode (No Run ID). ---", file=sys.stderr)
            else:
                print(f"--- [SDK] Initialized for Run ID: {self.run_id} ---", file=sys.stderr)
                self._metrics_queue = Queue()
                self._log_queue = Queue()
                self._progress_queue = Queue()
                self._stop_event = threading.Event()
                self._telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
                self._telemetry_thread.start()

    def _telemetry_worker(self):
        """Background worker to flush metrics, logs, progress, and poll commands."""
        stop_file = Path("STOP_REQUESTED")
        pause_file = Path("PAUSE_REQUESTED")

        # Increase interval to reduce server load
        polling_interval = 3.0

        while not self._stop_event.is_set():
            try:
                self._flush_metrics()
                self._flush_logs()
                self._flush_progress()
                
                cmd = self._poll_command()
                if cmd == "stop":
                    if not stop_file.exists():
                        stop_file.touch()
                        print(f"--- [SDK] Command Received: STOP. Creating signal file. ---", file=sys.stderr)
                        self.log_message("🛑 Received remote STOP command.\n")
                elif cmd == "pause":
                    if not pause_file.exists():
                        pause_file.touch()
                        print(f"--- [SDK] Command Received: PAUSE. Creating signal file. ---", file=sys.stderr)
                        self.log_message("⏸️ Received remote PAUSE command.\n")
                elif cmd == "resume":
                    if pause_file.exists():
                        pause_file.unlink()
                        print(f"--- [SDK] Command Received: RESUME. Removing signal file. ---", file=sys.stderr)
                        self.log_message("▶️ Received remote RESUME command.\n")
                
                time.sleep(polling_interval)
            except Exception as e:
                # Log error but don't crash thread
                # Reduce noise for connection refused if server is restarting
                err_str = str(e)
                if "Connection refused" in err_str:
                    print(f"[SDK-WARN] Connection refused (server down?). Retrying...", file=sys.stderr)
                else:
                    print(f"[SDK-ERR] Telemetry worker error: {e}", file=sys.stderr)
                
                time.sleep(5.0) # Backoff
        
        # Flush one last time on exit handled by shutdown() method explicitly

    def _poll_command(self):
        if not self.enabled or not self.run_id: return None
        try:
            # Increase timeout to 10s to handle server load spikes
            resp = requests.get(
                f"{self.api_endpoint}/api/telemetry/command/{self.run_id}", 
                headers=self.headers, 
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("command")
            elif resp.status_code != 200:
                # Log unexpected status codes (e.g. 500, 404) to help debug
                print(f"[SDK-ERR] Poll command failed. Status: {resp.status_code}", file=sys.stderr)
        except Exception as e:
            # Re-raise to be handled by worker loop backoff
            raise e
        return None

    def _flush_metrics(self):
        if self._metrics_queue.empty(): return
        batch = []
        while not self._metrics_queue.empty() and len(batch) < 100:
            try:
                batch.append(self._metrics_queue.get_nowait())
            except Empty:
                break
        if not batch: return
        payload = {"run_id": self.run_id, "metrics": batch}
        try:
            requests.post(f"{self.api_endpoint}/api/telemetry/metrics", json=payload, headers=self.headers, timeout=10)
        except Exception as e:
            print(f"[SDK-ERR] Metrics push failed: {e}", file=sys.stderr)

    def _flush_logs(self):
        while not self._log_queue.empty():
            try:
                msg = self._log_queue.get_nowait()
                payload = {"run_id": self.run_id, "message": msg}
                requests.post(f"{self.api_endpoint}/api/telemetry/log", json=payload, headers=self.headers, timeout=10)
            except Empty:
                break
            except Exception as e:
                print(f"[SDK-ERR] Log push failed: {e}", file=sys.stderr)
    
    def _flush_progress(self):
        latest_step = None
        while not self._progress_queue.empty():
            try:
                latest_step = self._progress_queue.get_nowait()
            except Empty:
                break
        if latest_step is not None:
            payload = {"run_id": self.run_id, "step": latest_step}
            try:
                requests.post(f"{self.api_endpoint}/api/telemetry/progress", json=payload, headers=self.headers, timeout=10)
            except Exception as e:
                print(f"[SDK-ERR] Progress push failed: {e}", file=sys.stderr)

    def shutdown(self):
        """Cleanly shuts down the client, flushing all pending data."""
        if hasattr(self, '_stop_event') and not self._stop_event.is_set():
            print("--- [SDK] Shutting down telemetry client... ---", file=sys.stderr)
            self._stop_event.set()
            if self._telemetry_thread.is_alive():
                self._telemetry_thread.join(timeout=3.0)
            
            # Explicit synchronous flush of any remaining items
            try:
                self._flush_metrics()
                self._flush_logs()
                self._flush_progress()
                print("--- [SDK] Telemetry flushed. ---", file=sys.stderr)
            except Exception as e:
                print(f"[SDK-ERR] Final flush error: {e}", file=sys.stderr)

    # --- Telemetry Methods ---

    def log_message(self, message: str):
        if self.enabled and self.run_id:
            self._log_queue.put(message)
        else:
            print(message, flush=True)

    def log_metric(self, name: str, step: int, value: float):
        if self.enabled and self.run_id:
            self._metrics_queue.put({"step": step, "name": name, "value": value})
            
    def log_progress(self, step: int):
        if self.enabled and self.run_id:
            self._progress_queue.put(step)
        
    def log_event(self, event_name: str, status: str = "info"):
        if self.enabled and self.run_id:
            print(f"--- [SDK] Event: {event_name} ({status}) ---", file=sys.stderr)
            # Send immediately via main thread for events, don't queue
            payload = {"event": event_name, "status": status}
            try:
                requests.post(f"{self.api_endpoint}/api/telemetry/event/{self.run_id}", json=payload, headers=self.headers, timeout=10)
            except Exception as e:
                print(f"[SDK-ERR] Event send failed: {e}", file=sys.stderr)
        else:
            print(f"[EVENT] {event_name} ({status})", flush=True)

    def update_total_steps(self, total_steps: int):
        if self.enabled and self.run_id:
            payload = {"run_id": self.run_id, "step": total_steps}
            try:
                requests.post(f"{self.api_endpoint}/api/telemetry/total_steps", json=payload, headers=self.headers, timeout=10)
            except Exception as e:
                print(f"[SDK-ERR] Total steps update failed: {e}", file=sys.stderr)

    # --- Data & Configuration ---

    def get_run_config(self) -> dict | None:
        if not self.enabled or not self.run_id: return None
        print(f"--- [SDK] Fetching configuration from API... ---", file=sys.stderr)
        endpoint = f"{self.api_endpoint}/api/runs/detail"
        try:
            response = requests.post(endpoint, json={"id": self.run_id}, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"--- [SDK] Configuration received. ---", file=sys.stderr)
            return data.get("config", {})
        except Exception as e:
            print(f"[SDK-ERR] Config fetch failed: {e}", file=sys.stderr)
            return None

    def upload_artifact(self, file_path: str, artifact_type: str = 'generic'):
        if not self.enabled or not self.run_id:
            print(f"[SDK] Upload skipped: SDK disabled.", file=sys.stderr)
            return
        
        print(f"--- [SDK] Uploading artifact: {os.path.basename(file_path)}... ---", file=sys.stderr)
        endpoint = f"{self.api_endpoint}/api/runs/{self.run_id}/artifacts"
        
        # Create headers without 'Content-Type' for multipart upload
        upload_headers = self.headers.copy()
        upload_headers.pop("Content-Type", None)

        try:
            with open(file_path, 'rb') as f:
                # Explicit MIME type 'application/octet-stream' to prevent 422 errors
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'artifact_type': artifact_type}
                resp = requests.post(endpoint, files=files, data=data, headers=upload_headers, timeout=600)
                
                if resp.status_code != 200:
                     print(f"--- [SDK-ERR] Backend returned {resp.status_code}: {resp.text} ---", file=sys.stderr)
                     resp.raise_for_status()

                print(f"--- [SDK] Upload success: {os.path.basename(file_path)} ---", file=sys.stderr)
        except Exception as e:
            # FATAL ERROR: Re-raise so the agent knows upload failed and can exit/retry
            print(f"[SDK-ERR] Upload failed for {file_path}: {e}", file=sys.stderr)
            raise e

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
            print(f"--- [SDK-ERR] {msg} ---", file=sys.stderr)
            if self.agent: self.agent.log(msg)
            return False
            
    def download_artifacts_for_run(self, source_run_id: UUID, destination_folder: Path) -> bool:
        print(f"--- [SDK] Fetching artifacts list for run {source_run_id}... ---", file=sys.stderr)
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

        if not self.enabled: return None
        
        endpoint = f"{self.api_endpoint}/api/data/training_set"
        try:
            if self.agent: self.agent.emit_event("TRAINING_DATA_REQUESTED")
            # Increase timeout for large data requests
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