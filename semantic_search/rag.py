#!/usr/bin/env python3
"""
MerkleDB RAG (Retrieval Augmented Generation)
Answer questions using MerkleDB semantic search + LLM.
"""

import os
import sys
from typing import List, Dict
from search import SemanticSearch

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "llama2")  # Model name
LLM_API_KEY = os.getenv("LLM_API_KEY", "")


class RAGSystem:
    def __init__(self):
        self.search_engine = SemanticSearch()
        self.llm_provider = LLM_PROVIDER
        self.llm_model = LLM_MODEL
        print(f"[RAG] RAG System")
        print(f"   LLM Provider: {self.llm_provider}")
        print(f"   LLM Model: {self.llm_model}\n")

    def retrieve_context(self, question: str, k: int = 3) -> tuple[List[Dict], str]:
        """Retrieve relevant documents and format as context."""
        results = self.search_engine.search(question, k=k)

        if not results:
            return [], ""

        # Format context from search results
        context_parts = []
        for i, result in enumerate(results, 1):
            doc_id = result.get("id", "unknown")
            score = result.get("score", 0.0)

            # Extract filename
            filename = doc_id.rsplit("_chunk_", 1)[0] if "_chunk_" in doc_id else doc_id

            context_parts.append(f"[Document {i}: {filename}, Relevance: {score:.2f}]")
            context_parts.append(f"ID: {doc_id}")
            context_parts.append("")  # Blank line

        context = "\n".join(context_parts)
        return results, context

    def generate_answer_ollama(self, question: str, context: str) -> str:
        """Generate answer using Ollama (local LLM)."""
        import requests

        prompt = f"""Based on the following context from MerkleDB documentation, answer the question.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}\nMake sure Ollama is running: https://ollama.ai"

    def generate_answer_openai(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            import openai
        except ImportError:
            return "Error: openai package not installed. Run: pip install openai"

        if not LLM_API_KEY:
            return "Error: LLM_API_KEY environment variable not set"

        openai.api_key = LLM_API_KEY

        prompt = f"""Based on the following context from MerkleDB documentation, answer the question.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}"""

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about MerkleDB based on provided documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {e}"

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using configured LLM provider."""
        if self.llm_provider == "ollama":
            return self.generate_answer_ollama(question, context)
        elif self.llm_provider == "openai":
            return self.generate_answer_openai(question, context)
        else:
            return f"Unsupported LLM provider: {self.llm_provider}"

    def answer_question(self, question: str, k: int = 3):
        """Complete RAG pipeline: retrieve + generate."""
        print(f"[?] Question: {question}\n")

        # Step 1: Retrieve context
        print("[*] Retrieving relevant documents...")
        results, context = self.retrieve_context(question, k=k)

        if not results:
            print("[ERROR] No relevant documents found.")
            return

        print(f"   Found {len(results)} relevant documents\n")

        # Display retrieved documents
        print("[*] Retrieved Context:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            doc_id = result.get("id", "unknown")
            score = result.get("score", 0.0)
            filename = doc_id.rsplit("_chunk_", 1)[0] if "_chunk_" in doc_id else doc_id
            print(f"  [{i}] {filename} (score: {score:.4f})")
        print("-" * 80 + "\n")

        # Step 2: Generate answer
        print("[RAG] Generating answer...")
        answer = self.generate_answer(question, context)

        print("\n[ANSWER] Answer:")
        print("=" * 80)
        print(answer)
        print("=" * 80 + "\n")

    def interactive_mode(self):
        """Interactive RAG Q&A."""
        print("[RAG] Interactive RAG Q&A System")
        print("   Ask questions about MerkleDB documentation")
        print("   Type 'quit' or 'exit' to stop\n")

        # Load model once
        self.search_engine.load_model()

        while True:
            try:
                question = input("\n[?] Question> ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("[*] Goodbye!")
                    break

                if not question:
                    continue

                self.answer_question(question)

            except KeyboardInterrupt:
                print("\n[*] Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MerkleDB RAG System")
    parser.add_argument("question", nargs="*", help="Question to answer")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Number of context documents (default: 3)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama", help="LLM provider")
    parser.add_argument("--model", help="LLM model name")

    args = parser.parse_args()

    # Override config from args
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    rag_system = RAGSystem()

    if args.interactive or not args.question:
        rag_system.interactive_mode()
    else:
        question = " ".join(args.question)
        rag_system.answer_question(question, k=args.top_k)


if __name__ == "__main__":
    main()
