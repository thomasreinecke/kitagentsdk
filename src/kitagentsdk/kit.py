# src/kitagentsdk/kit.py
import os
import requests
import logging
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
            # Use print for local runs as logger might not be configured by the agent script yet
            print("--- [SDK-WARN] KitClient is missing KIT_API_ENDPOINT or KIT_API_KEY. API calls will be disabled. Check .env file. ---")
            self.enabled = False
        else:
            self.enabled = True
            self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
            if not self.run_id:
                print("--- [SDK] KitClient initialized for local run (no KIT_RUN_ID detected). ---")

    def download_artifact(self, artifact_id: UUID, destination_path: str | Path) -> bool:
        """
        Downloads an artifact by its ID and saves it to a local path.
        """
        if not self.enabled:
            msg = "Cannot download artifact: KitClient is not enabled. Check .env file for KIT_API_ENDPOINT and KIT_API_KEY."
            # Use self.agent.log if available (in a kitexec run), otherwise print for local runs.
            if self.agent: self.agent.log(msg)
            else: print(msg)
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
            else: print(msg)
            if self.agent: self.agent.emit_event("ARTIFACT_DOWNLOAD_FAILED", "failure")
            return False
        except IOError as e:
            msg = f"Failed to write artifact to disk: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg)
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
            else: print(msg)
            return False

        if not artifacts:
            msg = f"No artifacts found for run {source_run_id}."
            if self.agent: self.agent.log(msg)
            else: print(msg)
            return False

        for artifact in artifacts:
            self.download_artifact(artifact['id'], destination_folder / artifact['filename'])
        return True

    def get_training_data(self, params: dict) -> dict | None:
        """
        Requests a fully prepared training dataset from the Kit backend.
        """
        if not self.enabled:
            msg = "Cannot get training data: KitClient is not enabled. Check .env file for KIT_API_ENDPOINT and KIT_API_KEY."
            if self.agent: self.agent.log(msg)
            else: print(msg)
            return None
        
        endpoint = f"{self.api_endpoint}/api/data/training_set"
        
        try:
            if self.agent: self.agent.emit_event("TRAINING_DATA_REQUESTED")
            response = requests.post(endpoint, json=params, headers=self.headers, timeout=300)
            response.raise_for_status()
            if self.agent: self.agent.emit_event("TRAINING_DATA_RECEIVED", "success")
            return response.json()
        except requests.RequestException as e:
            msg = f"Failed to get training data: {e}"
            if self.agent: self.agent.log(msg)
            else: print(msg)
            if self.agent: self.agent.emit_event("TRAINING_DATA_FAILED", "failure")
            
            # --- THIS IS THE FIX (Part 3) ---
            # Try to parse the JSON error response from the backend for a clearer message.
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", e.response.text)
                    err_body = f"Response body: {error_detail}"
                    if self.agent: self.agent.log(err_body)
                    else: print(err_body)
                except Exception:
                    err_body = f"Response body: {e.response.text}"
                    if self.agent: self.agent.log(err_body)
                    else: print(err_body)
            return None