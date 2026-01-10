#!/usr/bin/env python3
"""
MerkleDB Semantic Search - Document Ingestion
Indexes markdown documentation and code into MerkleDB for semantic search.
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Configuration
MERKLEDB_URL = os.getenv("MERKLEDB_URL", "http://localhost:4000")
MERKLEDB_COLLECTION = os.getenv("MERKLEDB_COLLECTION", "docs")
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions

class DocumentIngester:
    def __init__(self, merkledb_url: str = MERKLEDB_URL):
        self.merkledb_url = merkledb_url
        self.collection = MERKLEDB_COLLECTION
        self.model = None
        print(f"[*] Initializing Document Ingester")
        print(f"   MerkleDB: {self.merkledb_url}")
        print(f"   Collection: {self.collection}")

    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            print(f"[*] Loading embedding model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME)
            print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        return self.model

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks.
        Tries to break on sentence boundaries when possible.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) > 20]  # Filter very short chunks

    def read_markdown_file(self, filepath: Path) -> Tuple[str, Dict]:
        """Read markdown file and extract metadata."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else filepath.stem

        metadata = {
            "source": str(filepath),
            "filename": filepath.name,
            "title": title,
            "type": "documentation"
        }

        return content, metadata

    def find_documents(self, root_dir: Path) -> List[Path]:
        """Find all markdown files to index."""
        patterns = ["*.md", "*.markdown"]
        exclude_dirs = {"node_modules", "deps", ".git", "_build", "target"}

        docs = []
        for pattern in patterns:
            for filepath in root_dir.rglob(pattern):
                # Skip excluded directories
                if any(excluded in filepath.parts for excluded in exclude_dirs):
                    continue
                docs.append(filepath)

        return sorted(docs)

    def create_collection(self):
        """Create MerkleDB collection if it doesn't exist."""
        dimension = self.model.get_sentence_embedding_dimension()
        url = f"{self.merkledb_url}/v1/collections/{self.collection}"

        payload = {
            "dimension": dimension,
            "precision": "f32"
        }

        response = requests.post(url, json=payload)
        if response.status_code == 201:
            print(f"[OK] Created collection '{self.collection}' (dimension: {dimension})")
        elif response.status_code == 409:
            print(f"[INFO] Collection '{self.collection}' already exists")
        else:
            print(f"[WARN] Collection creation: {response.status_code} - {response.text}")

    def insert_batch(self, vectors: List[Dict]) -> bool:
        """Insert batch of vectors into MerkleDB."""
        url = f"{self.merkledb_url}/v1/{self.collection}/vectors"

        # Convert to MerkleDB format
        payload = []
        for vec in vectors:
            payload.append({
                "id": vec["id"],
                "vector": vec["embedding"].tolist(),
                "metadata": vec["metadata"]
            })

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True
        else:
            print(f"[ERROR] Insert failed: {response.status_code} - {response.text}")
            return False

    def ingest_documents(self, root_dir: Path):
        """Main ingestion pipeline."""
        print(f"\n[*] Finding documents in: {root_dir}")

        # Load model
        model = self.load_model()

        # Create collection
        self.create_collection()

        # Find all documents
        documents = self.find_documents(root_dir)
        print(f"   Found {len(documents)} documents")

        if not documents:
            print("[WARN] No documents found!")
            return

        total_chunks = 0
        batch = []
        batch_size = 10  # Insert 10 chunks at a time

        print(f"\n[*] Processing documents...")

        for doc_path in documents:
            print(f"   [*] {doc_path.name}")

            # Read document
            content, metadata = self.read_markdown_file(doc_path)

            # Chunk document
            chunks = self.chunk_text(content)
            print(f"      -> {len(chunks)} chunks")

            # Generate embeddings and prepare for insertion
            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk, normalize_embeddings=True)

                chunk_id = f"{doc_path.stem}_chunk_{i}"
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_text": chunk[:200]  # Store preview
                }

                batch.append({
                    "id": chunk_id,
                    "embedding": embedding,
                    "metadata": chunk_metadata
                })

                total_chunks += 1

                # Insert batch when full
                if len(batch) >= batch_size:
                    if self.insert_batch(batch):
                        print(f"      [OK] Inserted batch ({len(batch)} chunks)")
                    batch = []

        # Insert remaining
        if batch:
            if self.insert_batch(batch):
                print(f"   [OK] Inserted final batch ({len(batch)} chunks)")

        print(f"\n[*] Ingestion complete!")
        print(f"   Total chunks indexed: {total_chunks}")
        print(f"   Documents processed: {len(documents)}")


def main():
    import sys

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent  # merkle_db root

    if not root_dir.exists():
        print(f"[ERROR] Directory not found: {root_dir}")
        sys.exit(1)

    ingester = DocumentIngester()
    ingester.ingest_documents(root_dir)


if __name__ == "__main__":
    main()
