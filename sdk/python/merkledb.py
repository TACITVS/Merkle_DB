import requests
import json
from typing import List, Dict, Union, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class MerkleDb:
    """
    Python SDK for MerkleDb Vector Database.
    """

    def __init__(self, url: str = "http://localhost:4000", api_token: str = "ABC-123"):
        self.url = url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def list_collections(self) -> List[str]:
        """List all available collections."""
        resp = requests.get(f"{self.url}/v1/collections", headers=self.headers)
        resp.raise_for_status()
        return resp.json().get("collections", [])

    def create_collection(self, name: str, dim: int, precision: str = "f32"):
        """Create a new collection."""
        data = {"dim": dim, "precision": precision}
        resp = requests.post(f"{self.url}/v1/collections/{name}", headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json()

    def drop_collection(self, name: str):
        """Drop a collection."""
        resp = requests.delete(f"{self.url}/v1/collections/{name}", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def insert(self, collection: str, items: List[Dict[str, Any]]):
        """
        Insert items into a collection.
        Items should be a list of dicts:
        [{"id": "vec1", "vector": [0.1, ...], "metadata": {"key": "val"}}]
        OR [{"id": "vec1", "text": "semantic text", "metadata": {...}}]
        """
        # Convert numpy arrays to lists if present
        if HAS_NUMPY:
            for item in items:
                if "vector" in item and isinstance(item["vector"], np.ndarray):
                    item["vector"] = item["vector"].tolist()

        resp = requests.post(f"{self.url}/v1/{collection}/vectors", headers=self.headers, json=items)
        resp.raise_for_status()
        return resp.json()

    def search(self, 
               collection: str, 
               vector: Optional[Union[List[float], Any]] = None, 
               text: Optional[str] = None, 
               k: int = 10, 
               threshold: float = 0.0):
        """
        Search for nearest neighbors.
        Provide either 'vector' (list or numpy array) or 'text' (for semantic search).
        """
        data = {"k": k, "threshold": threshold}
        
        if vector is not None:
            if HAS_NUMPY and isinstance(vector, np.ndarray):
                vector = vector.tolist()
            data["vector"] = vector
        elif text is not None:
            data["text"] = text
        else:
            raise ValueError("Must provide either 'vector' or 'text'")

        resp = requests.post(f"{self.url}/v1/{collection}/search", headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def checkpoint(self, collection: str):
        """Force a disk checkpoint for a collection."""
        resp = requests.post(f"{self.url}/v1/{collection}/checkpoint", headers=self.headers)
        resp.raise_for_status()
        return resp.json()
