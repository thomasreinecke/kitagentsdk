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
            if hasattr(e, 'response') and e.response is not None:
                err_body = f"Response body: {e.response.text}"
                if self.agent: self.agent.log(err_body)
                else: print(err_body)
            return None