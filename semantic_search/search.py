#!/usr/bin/env python3
"""
MerkleDB Semantic Search - Query Interface
Search indexed documents using natural language queries.
"""

import os
import sys
import json
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Configuration
MERKLEDB_URL = os.getenv("MERKLEDB_URL", "http://localhost:4000")
MERKLEDB_COLLECTION = os.getenv("MERKLEDB_COLLECTION", "docs")
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self, merkledb_url: str = MERKLEDB_URL):
        self.merkledb_url = merkledb_url
        self.collection = MERKLEDB_COLLECTION
        self.model = None
        print(f"[SEARCH] Semantic Search Engine")
        print(f"   MerkleDB: {self.merkledb_url}")
        print(f"   Collection: {self.collection}\n")

    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            print(f"[*] Loading model: {MODEL_NAME}...")
            self.model = SentenceTransformer(MODEL_NAME)
        return self.model

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for documents matching the query.

        Args:
            query: Natural language search query
            k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        # Load model
        model = self.load_model()

        # Generate query embedding
        query_embedding = model.encode(query, normalize_embeddings=True)

        # Search MerkleDB
        url = f"{self.merkledb_url}/v1/{self.collection}/search"
        payload = {
            "vector": query_embedding.tolist(),
            "k": k
        }

        response = requests.post(url, json=payload)

        if response.status_code != 200:
            print(f"[ERROR] Search failed: {response.status_code} - {response.text}")
            return []

        results = response.json().get("results", [])
        return results

    def display_results(self, query: str, results: List[Dict]):
        """Display search results in a user-friendly format."""
        print(f"[SEARCH] Query: \"{query}\"")
        print(f"   Found {len(results)} results\n")

        if not results:
            print("   No results found.")
            return

        print("=" * 80)

        for i, result in enumerate(results, 1):
            doc_id = result.get("id", "unknown")
            score = result.get("score", 0.0)

            # Parse metadata from ID (format: filename_chunk_N)
            parts = doc_id.rsplit("_chunk_", 1)
            if len(parts) == 2:
                filename = parts[0]
                chunk_idx = parts[1]
            else:
                filename = doc_id
                chunk_idx = "?"

            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Document: {filename}")
            print(f"    Chunk: #{chunk_idx}")
            print(f"    ID: {doc_id}")

        print("\n" + "=" * 80)

    def interactive_mode(self):
        """Interactive search shell."""
        print("[*] Interactive Semantic Search")
        print("   Type your query and press Enter")
        print("   Type 'quit' or 'exit' to stop\n")

        # Load model once
        self.load_model()

        while True:
            try:
                query = input("\n[SEARCH] Search> ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("[*] Goodbye!")
                    break

                if not query:
                    continue

                # Parse k parameter if provided
                k = 5
                if query.startswith("k="):
                    parts = query.split(maxsplit=1)
                    k = int(parts[0].split("=")[1])
                    query = parts[1] if len(parts) > 1 else ""
                    if not query:
                        continue

                # Search
                results = self.search(query, k=k)
                self.display_results(query, results)

            except KeyboardInterrupt:
                print("\n[*] Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MerkleDB Semantic Search")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    search_engine = SemanticSearch()

    if args.interactive or not args.query:
        # Interactive mode
        search_engine.interactive_mode()
    else:
        # Single query mode
        query = " ".join(args.query)
        results = search_engine.search(query, k=args.top_k)
        search_engine.display_results(query, results)


if __name__ == "__main__":
    main()
