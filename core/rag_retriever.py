import json
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


class RagRetriever:
    """FAISS + sentence-transformers based RAG for PlantUML examples."""

    def __init__(
            self,
            faiss_index_path: str,
            faiss_meta_path: str,
            embed_model_name: str = "nomic-embed-text",
    ) -> None:
        self.index = faiss.read_index(faiss_index_path)
        with open(faiss_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # meta may be a list or dict; normalize to list indexed by int
        if isinstance(meta, list):
            self.meta: List[Dict[str, Any]] = meta
        elif isinstance(meta, dict):
            # Check if it has a "documents" array (common format)
            if "documents" in meta:
                self.meta = meta["documents"]
            else:
                # assume keys are str(int)
                max_idx = max(int(k) for k in meta.keys())
                self.meta = [{} for _ in range(max_idx + 1)]
                for k, v in meta.items():
                    self.meta[int(k)] = v
        else:
            raise ValueError("Unsupported meta JSON structure")

        # Normalize field names across different metadata formats
        self.meta = self._normalize_metadata(self.meta)

        self.model = SentenceTransformer(embed_model_name)
        self.dim = self.index.d

    def _normalize_metadata(self, meta_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize field names to a consistent format.
        
        Handles different metadata structures:
        - diagram_type / type
        - output / plantuml
        - instruction / description
        """
        normalized = []
        
        for entry in meta_list:
            norm_entry = dict(entry)  # Copy
            
            # Normalize diagram type field
            if "diagram_type" in norm_entry and "type" not in norm_entry:
                norm_entry["type"] = norm_entry["diagram_type"]
            
            # Normalize PlantUML code field
            if "output" in norm_entry and "plantuml" not in norm_entry:
                norm_entry["plantuml"] = norm_entry["output"]
            
            # Normalize description field
            if "instruction" in norm_entry and "description" not in norm_entry:
                norm_entry["description"] = norm_entry["instruction"]
            
            normalized.append(norm_entry)
        
        return normalized

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        if vec.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {vec.shape[1]}, index expects {self.dim}"
            )
        return vec.astype("float32")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        emb = self.embed(query)
        scores, idxs = self.index.search(emb, top_k)
        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            entry = dict(self.meta[idx])
            entry["score"] = float(score)
            out.append(entry)
        return out
