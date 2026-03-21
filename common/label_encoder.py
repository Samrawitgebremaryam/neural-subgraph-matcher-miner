"""
Encodes node and edge type labels (e.g. "Gene", "binds") into fixed-size
embedding vectors for use in heterogeneous graph neural networks.

Uses sentence-transformers (all-MiniLM-L6-v2) if available,
otherwise falls back to a stable hash-based encoding.
"""

import hashlib
import torch
import numpy as np

LABEL_EMB_DIM = 384


class UniversalLabelEncoder:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.cache = {}
        self.model = None
        self.dim = LABEL_EMB_DIM

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[LabelEncoder] Loaded sentence-transformer: {model_name}")
        except Exception as e:
            print(
                f"[LabelEncoder] Could not load sentence-transformers "
                f"({type(e).__name__}: {e}). "
                f"Using hash-based fallback encoding."
            )

    def encode(self, label: str) -> torch.Tensor:
        """Encode a single string label into a tensor of shape (dim,)."""
        if label in self.cache:
            return self.cache[label]
        if self.model is not None:
            emb = self.model.encode(label, convert_to_tensor=True).detach()
        else:
            emb = self._hash_encode(label)
        self.cache[label] = emb
        return emb

    def _hash_encode(self, label: str) -> torch.Tensor:
        """Deterministic hash-based fallback using hashlib.md5."""
        digest = hashlib.md5(label.encode("utf-8")).hexdigest()
        seed = int(digest, 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return torch.tensor(vec)

    def encode_with_fallback(self, label: str) -> torch.Tensor:
        """Returns tensor([1.0]) for empty/None labels, full embedding otherwise."""
        if not label or label.strip() == "":
            return torch.tensor([1.0])
        return self.encode(str(label))


_global_encoder = None


def get_label_encoder() -> UniversalLabelEncoder:
    """Get or create the global label encoder instance."""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = UniversalLabelEncoder()
    return _global_encoder


def _run_tests():
    import sys
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS {name}")
            passed += 1
        else:
            print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    print("\n=== Test 1: Stable hash across separate instances ===")
    enc1 = UniversalLabelEncoder()
    enc2 = UniversalLabelEncoder()
    for label in ["Gene", "Protein", "Drug", "Disease", "binds", "regulates"]:
        v1 = enc1._hash_encode(label)
        v2 = enc2._hash_encode(label)
        check(f"stable hash for '{label}'", torch.allclose(v1, v2))
    check(
        "different labels produce different vectors",
        not torch.allclose(enc1._hash_encode("Gene"), enc1._hash_encode("Protein"))
    )
    check(
        "hash vector is unit normalized",
        abs(torch.norm(enc1._hash_encode("TestLabel")).item() - 1.0) < 1e-5
    )

    print("\n=== Test 2: Output shape and dtype ===")
    enc = UniversalLabelEncoder()
    emb = enc.encode("Gene")
    check("encode() returns torch.Tensor", isinstance(emb, torch.Tensor))
    check("encode() shape is (384,)", emb.shape == (LABEL_EMB_DIM,), f"got {emb.shape}")
    check("encode() dtype is float32", emb.dtype == torch.float32, f"got {emb.dtype}")
    check("no NaN in embedding", not torch.isnan(emb).any())
    for empty in [None, "", " "]:
        result = enc.encode_with_fallback(empty)
        check(f"encode_with_fallback({repr(empty)}) → tensor([1.0])",
              torch.equal(result, torch.tensor([1.0])))
    result = enc.encode_with_fallback("Drug")
    check("encode_with_fallback valid label → full embedding",
          result.shape == (LABEL_EMB_DIM,))
    e1 = enc.encode("cached_label")
    e2 = enc.encode("cached_label")
    check("cache returns same tensor object", e1 is e2)

    print("\n=== Test 3: Fallback when sentence-transformers unavailable ===")
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = None
    try:
        enc_fb = UniversalLabelEncoder()
        check("model is None when package unavailable", enc_fb.model is None)
        emb = enc_fb.encode("Disease")
        check("fallback shape is (384,)", emb.shape == (LABEL_EMB_DIM,))
        check("fallback dtype is float32", emb.dtype == torch.float32)
        check("fallback has no NaN", not torch.isnan(emb).any())
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original

    enc_a = get_label_encoder()
    enc_b = get_label_encoder()
    check("get_label_encoder() is singleton", enc_a is enc_b)

    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    _run_tests()
