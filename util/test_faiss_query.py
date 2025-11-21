#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict

import faiss
import numpy as np
import requests


def ollama_embed(texts: List[str], embed_model: str, ollama_url: str) -> np.ndarray:
    resp = requests.post(
        f"{ollama_url}/api/embed",
        json={"model": embed_model, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if embeddings is None:
        raise RuntimeError(f"Ollama /api/embed returned no 'embeddings' field: {data}")
    return np.array(embeddings, dtype="float32")


def main():
    parser = argparse.ArgumentParser(description="Test FAISS RAG retrieval using Ollama embeddings.")
    parser.add_argument(
        "query",
        nargs="?",
        default="activity diagram with decisions and branches",
        help="Test query text (default: 'activity diagram with decisions and branches')",
    )
    parser.add_argument(
        "--faiss-index",
        default=os.environ.get("RAG_FAISS_INDEX", "rag/faiss.index"),
        help="Path to FAISS index file (default: RAG_FAISS_INDEX or rag/faiss.index)",
    )
    parser.add_argument(
        "--faiss-meta",
        default=os.environ.get("RAG_FAISS_META", "rag/faiss_meta.json"),
        help="Path to FAISS metadata JSON (default: RAG_FAISS_META or rag/faiss_meta.json)",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text"),
        help="Embedding model to use with Ollama (default: RAG_EMBED_MODEL or nomic-embed-text)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL (default: OLLAMA_URL or http://localhost:11434)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("RAG_TOP_K", "5")),
        help="Number of nearest neighbors to show (default: 5 or RAG_TOP_K)",
    )

    args = parser.parse_args()

    print(f"[test_faiss_query] Loading FAISS index from {args.faiss_index}")
    index = faiss.read_index(args.faiss_index)

    print(f"[test_faiss_query] Loading metadata from {args.faiss_meta}")
    with open(args.faiss_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs: List[Dict] = meta["documents"]

    print(f"[test_faiss_query] Query: {args.query}")
    q_emb = ollama_embed([args.query], args.embed_model, args.ollama_url)
    scores, indices = index.search(q_emb, args.top_k)

    print("\n[test_faiss_query] Top results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        doc = docs[idx]
        instr = doc.get("instruction", "")
        plantuml = doc.get("output") or doc.get("plantuml", "")
        diagram_type = doc.get("diagram_type", "<unknown>")
        print("---")
        print(f"Rank: {rank}")
        print(f"Score: {score}")
        print(f"Diagram type: {diagram_type}")
        print(f"Instruction: {instr}")
        print("PlantUML snippet:")
        print(plantuml[:400])
        if len(plantuml) > 400:
            print("... [truncated]")


if __name__ == "__main__":
    main()

