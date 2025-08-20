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
        self.run_id = os.getenv("KIT_RUN_ID")
        
        if not all([self.api_endpoint, self.api_key, self.run_id]):
            logger.warning("KitClient initialized without full API credentials. API calls will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
            logger.info(f"KitClient initialized for Run ID: {self.run_id}")

    def download_artifact(self, artifact_id: UUID, destination_path: str | Path) -> bool:
        """
        Downloads an artifact by its ID and saves it to a local path.

        Args:
            artifact_id: The specific UUID of the artifact to download.
            destination_path: The local path (including filename) to save the artifact.

        Returns:
            True if the download was successful, False otherwise.
        """
        if not self.enabled:
            logger.error("Cannot download artifact: KitClient is not enabled.")
            return False

        endpoint = f"{self.api_endpoint}/api/artifacts/{artifact_id}/download"
        
        try:
            print(f"Downloading artifact {artifact_id} from {endpoint}...")
            with requests.get(endpoint, headers={"X-API-KEY": self.api_key}, stream=True) as r:
                r.raise_for_status()
                with open(destination_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            print(f"✅ Successfully downloaded artifact to '{destination_path}'")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to download artifact: {e}")
            return False
        except IOError as e:
            logger.error(f"Failed to write artifact to disk: {e}")
            return False

    def get_training_data(self, params: dict) -> dict | None:
        """
        Requests a fully prepared training dataset from the Kit backend.

        Args:
            params: A dictionary of parameters for the data request, 
                    e.g., {"symbol": "YM", "start_date": "2020-01-01", ...}

        Returns:
            A dictionary containing the dataframes if successful, None otherwise.
        """
        if not self.enabled:
            logger.error("Cannot get training data: KitClient is not enabled.")
            return None
        
        endpoint = f"{self.api_endpoint}/api/data/training_set"
        
        try:
            print(f"Requesting training data with params: {params}")
            response = requests.post(endpoint, json=params, headers=self.headers, timeout=300) # 5 min timeout for large data
            response.raise_for_status()
            print("✅ Successfully received training data.")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get training data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            return None