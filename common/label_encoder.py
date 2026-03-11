"""
Phase 1: Universal Label Encoder
Encodes node/edge type labels (e.g. "Gene", "binds") into semantic embeddings.
Falls back to a simple hash-based encoding if sentence-transformers is not available.
"""

import torch
import numpy as np

# Embedding dimension when using sentence-transformers
LABEL_EMB_DIM = 384

class UniversalLabelEncoder:
    """
    Encodes any string label into a fixed-size embedding vector.
    Uses sentence-transformers (all-MiniLM-L6-v2) if available,
    otherwise falls back to a deterministic hash-based encoding.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.cache = {}  # cache so we don't re-encode the same label
        self.model = None
        self.dim = LABEL_EMB_DIM

        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # Freeze all parameters - we don't want to fine-tune the language model
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[LabelEncoder] Loaded sentence-transformer: {model_name}")
        except ImportError:
            print("[LabelEncoder] sentence-transformers not installed. "
                  "Using hash-based fallback encoding.")

    def encode(self, label: str) -> torch.Tensor:
        """
        Encode a single string label into a tensor of shape (dim,).
        Results are cached so repeated calls are free.
        """
        if label in self.cache:
            return self.cache[label]

        if self.model is not None:
            # Use sentence-transformers
            emb = self.model.encode(label, convert_to_tensor=True)
            emb = emb.detach()
        else:
            # Fallback: deterministic hash-based encoding
            emb = self._hash_encode(label)

        self.cache[label] = emb
        return emb

    def _hash_encode(self, label: str) -> torch.Tensor:
        """
        Deterministic fallback encoding using a seeded random vector.
        Same label always produces the same vector.
        """
        seed = hash(label) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        # Normalize to unit length
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return torch.tensor(vec)

    def encode_with_fallback(self, label: str, fallback_dim: int = 1) -> torch.Tensor:
        """
        If label is None or empty, return the original fallback tensor([1.0]).
        Otherwise return the full semantic embedding.
        Used during the transition so existing code doesn't break.
        """
        if not label or label.strip() == "":
            return torch.tensor([1.0])
        return self.encode(str(label))


# Global singleton instance so we don't reload the model multiple times
_global_encoder = None

def get_label_encoder() -> UniversalLabelEncoder:
    """Get or create the global label encoder instance."""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = UniversalLabelEncoder()
    return _global_encoder