# src/kitagentsdk/kit.py
import os
import requests
import json
import logging
import sys
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

    def download_artifact(self, artifact_id: UUID, destination_path: str | Path) -> bool:
        """
        Downloads an artifact by its ID and saves it to a local path.
        """
        if not self.enabled:
            msg = "Cannot download artifact: KitClient is not enabled. Check .env file for KIT_API_ENDPOINT and KIT_API_KEY."
            # Use self.agent.log if available (in a kitexec run), otherwise print for local runs.
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            return False

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
        except requests.RequestException as e:
            msg = f"Failed to download artifact: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            if self.agent: self.agent.emit_event("ARTIFACT_DOWNLOAD_FAILED", "failure")
            return False
        except IOError as e:
            msg = f"Failed to write artifact to disk: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            if self.agent: self.agent.emit_event("ARTIFACT_DOWNLOAD_FAILED", "failure")
            return False
            
    def download_artifacts_for_run(self, source_run_id: UUID, destination_folder: Path) -> bool:
        """Finds and downloads all artifacts for a given run ID."""
        list_endpoint = f"{self.api_endpoint}/api/runs/{source_run_id}/artifacts/list"
        try:
            list_response = requests.post(list_endpoint, headers=self.headers, timeout=10)
            list_response.raise_for_status()
            artifacts = list_response.json()
        except requests.RequestException as e:
            msg = f"Failed to list artifacts for run {source_run_id}: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            return False

        if not artifacts:
            msg = f"No artifacts found for run {source_run_id}."
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            return False

        for artifact in artifacts:
            self.download_artifact(artifact['id'], destination_folder / artifact['filename'])
        return True

    def get_training_data(self, params: dict) -> dict | None:
        """
        Requests a fully prepared training dataset.
        Checks for a locally injected file first (provided by kitexec cache), 
        otherwise calls the Kit backend.
        """
        # 1. Check for local data injection (Simulation Optimization)
        local_data_path = os.getenv("KIT_LOCAL_DATA_PATH")
        if local_data_path and os.path.exists(local_data_path):
            msg = f"--- [SDK] Using locally injected data from {local_data_path} ---"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            try:
                with open(local_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # --- DEBUG VERBOSITY (Local) ---
                print("\n=== [SDK] Local Data Inspection ===", file=sys.stderr)
                for scale, content in data.items():
                    if content and isinstance(content, str) and content != "null":
                        try:
                            parsed = json.loads(content)
                            cols = parsed.get('columns', [])
                            rows = parsed.get('data', [])
                            print(f"Scale '{scale}': {len(rows)} rows | Cols: {cols}", file=sys.stderr)
                            if rows:
                                print(f"  First Row: {rows[0]}", file=sys.stderr)
                        except Exception:
                            pass
                print("===================================\n", file=sys.stderr)
                # ------------------------------
                return data
            except Exception as e:
                err_msg = f"Failed to load local data file: {e}. Falling back to API."
                if self.agent: self.agent.log(err_msg)
                else: print(err_msg, file=sys.stderr)

        # 2. Fallback to API Call
        if not self.enabled:
            msg = "Cannot get training data: KitClient is not enabled. Check .env file for KIT_API_ENDPOINT and KIT_API_KEY."
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            return None
        
        endpoint = f"{self.api_endpoint}/api/data/training_set"
        
        try:
            if self.agent: self.agent.emit_event("TRAINING_DATA_REQUESTED")
            response = requests.post(endpoint, json=params, headers=self.headers, timeout=300)
            response.raise_for_status()
            if self.agent: self.agent.emit_event("TRAINING_DATA_RECEIVED", "success")
            
            data = response.json()
            
            # --- DEBUG VERBOSITY (API) ---
            print("\n=== [SDK] API Data Inspection ===", file=sys.stderr)
            for scale, content in data.items():
                if content and isinstance(content, str) and content != "null":
                    try:
                        parsed = json.loads(content)
                        cols = parsed.get('columns', [])
                        rows = parsed.get('data', [])
                        print(f"Scale '{scale}': {len(rows)} rows | Cols: {cols}", file=sys.stderr)
                        if rows:
                            print(f"  First Row: {rows[0]}", file=sys.stderr)
                    except Exception:
                        pass
            print("=================================\n", file=sys.stderr)
            # ---------------------------
            
            return data
        except requests.RequestException as e:
            msg = f"Failed to get training data: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg, file=sys.stderr)
            if self.agent: self.agent.emit_event("TRAINING_DATA_FAILED", "failure")
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", e.response.text)
                    err_body = f"Response body: {error_detail}"
                    if self.agent: self.agent.log(err_body)
                    else: print(err_body, file=sys.stderr)
                except Exception:
                    err_body = f"Response body: {e.response.text}"
                    if self.agent: self.agent.log(err_body)
                    else: print(err_body, file=sys.stderr)
            return None