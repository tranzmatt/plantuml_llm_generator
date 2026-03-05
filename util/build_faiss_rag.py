#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict

import faiss
import numpy as np
import requests


def load_corpus(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: {path}")
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    if not docs:
        raise ValueError(f"Corpus file {path} is empty.")
    return docs


def extract_plantuml_text(doc: Dict) -> str:
    """
    Support Alpaca-style:
      { "instruction": "...", "input": "...", "output": "@startuml ... @enduml" }
    and older plantuml-specific:
      { "plantuml": "@startuml ... @enduml", ... }
    """
    if "output" in doc and doc["output"]:
        return doc["output"]
    if "plantuml" in doc and doc["plantuml"]:
        return doc["plantuml"]
    raise KeyError(f"Document missing PlantUML text: {doc}")


def ollama_embed(texts: List[str], embed_model: str, ollama_url: str) -> np.ndarray:
    """
    Use Ollama /api/embed to get embeddings.
    docs: https://docs.ollama.com/capabilities/embeddings 
    """
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
    parser = argparse.ArgumentParser(
        description="Build FAISS RAG index from an Alpaca-format PlantUML corpus using Ollama embeddings."
    )
    parser.add_argument(
        "--corpus",
        default=os.environ.get("RAG_CORPUS", "rag/corpus.jsonl"),
        help="Path to Alpaca JSONL corpus (default: RAG_CORPUS or rag/corpus.jsonl)",
    )
    parser.add_argument(
        "--faiss-index",
        default=os.environ.get("RAG_FAISS_INDEX", ""),
        help="Output FAISS index path (default: alongside corpus as faiss.index)",
    )
    parser.add_argument(
        "--faiss-meta",
        default=os.environ.get("RAG_FAISS_META", ""),
        help="Output FAISS metadata JSON path (default: alongside corpus as faiss_meta.json)",
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

    args = parser.parse_args()

    corpus_path = args.corpus
    embed_model = args.embed_model
    ollama_url = args.ollama_url

    base_dir = os.path.dirname(os.path.abspath(corpus_path)) or "."
    faiss_index_path = args.faiss_index or os.path.join(base_dir, "faiss.index")
    faiss_meta_path = args.faiss_meta or os.path.join(base_dir, "faiss_meta.json")

    os.makedirs(base_dir, exist_ok=True)

    print(f"[build_faiss_rag] Loading corpus from {corpus_path}")
    corpus = load_corpus(corpus_path)

    texts = [extract_plantuml_text(d) for d in corpus]
    print(f"[build_faiss_rag] Loaded {len(texts)} documents")

    print(f"[build_faiss_rag] Requesting embeddings from Ollama at {ollama_url} using model '{embed_model}'")
    embeddings = ollama_embed(texts, embed_model, ollama_url)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count {len(embeddings)} does not match text count {len(texts)}"
        )

    dim = embeddings.shape[1]
    print(f"[build_faiss_rag] Building FAISS IndexFlatIP with dimension {dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[build_faiss_rag] Writing FAISS index to {faiss_index_path}")
    faiss.write_index(index, faiss_index_path)

    meta = {
        "embed_model": embed_model,
        "corpus_path": os.path.abspath(corpus_path),
        "documents": corpus,
    }

    print(f"[build_faiss_rag] Writing metadata to {faiss_meta_path}")
    with open(faiss_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[build_faiss_rag] Done.")


if __name__ == "__main__":
    main()

