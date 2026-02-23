"""
MinHash-LSH deduplication for legal documents.

Uses locality-sensitive hashing to detect near-duplicate content
efficiently, even at scale (100K+ documents).
"""

import hashlib
import re
import struct
from typing import Optional, Set

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


class MinHash:
    """MinHash signature for a document."""

    __slots__ = ("num_perm", "hashvalues")

    def __init__(self, num_perm: int = 128) -> None:
        self.num_perm = num_perm
        self.hashvalues = [_MAX_HASH] * num_perm

    def update(self, token: str) -> None:
        h = _murmurhash32(token)
        for i in range(self.num_perm):
            a = _HASH_COEFFS_A[i % len(_HASH_COEFFS_A)]
            b = _HASH_COEFFS_B[i % len(_HASH_COEFFS_B)]
            val = ((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH
            if val < self.hashvalues[i]:
                self.hashvalues[i] = val

    def jaccard(self, other: "MinHash") -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        matches = sum(1 for a, b in zip(self.hashvalues, other.hashvalues) if a == b)
        return matches / self.num_perm

    def signature(self) -> str:
        """Compact hex signature for storage."""
        return hashlib.md5(
            struct.pack(f">{self.num_perm}I", *self.hashvalues)
        ).hexdigest()


def _murmurhash32(key: str) -> int:
    """Simple 32-bit hash for string tokens."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) & _MAX_HASH
    return h


_HASH_COEFFS_A = [
    (i * 0x5BD1E995 + 0x1B873593) & _MAX_HASH for i in range(256)
]
_HASH_COEFFS_B = [
    (i * 0xCC9E2D51 + 0xE6546B64) & _MAX_HASH for i in range(256)
]


class LSHIndex:
    """Locality-Sensitive Hashing index for fast approximate nearest-neighbor."""

    def __init__(self, num_perm: int = 128, num_bands: int = 16) -> None:
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self._buckets: list[dict[int, set[str]]] = [
            {} for _ in range(num_bands)
        ]
        self._signatures: dict[str, MinHash] = {}

    def insert(self, doc_id: str, minhash: MinHash) -> None:
        self._signatures[doc_id] = minhash
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(minhash.hashvalues[start:end]))
            bucket = self._buckets[band_idx]
            if band_hash not in bucket:
                bucket[band_hash] = set()
            bucket[band_hash].add(doc_id)

    def query(self, minhash: MinHash) -> set[str]:
        """Find candidate duplicates (may include false positives)."""
        candidates: set[str] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(minhash.hashvalues[start:end]))
            bucket = self._buckets[band_idx]
            if band_hash in bucket:
                candidates.update(bucket[band_hash])
        return candidates

    def query_with_scores(
        self, minhash: MinHash, threshold: float = 0.8
    ) -> list[tuple[str, float]]:
        """Find duplicates above Jaccard threshold, with scores."""
        candidates = self.query(minhash)
        results: list[tuple[str, float]] = []
        for doc_id in candidates:
            sig = self._signatures.get(doc_id)
            if sig is None:
                continue
            sim = minhash.jaccard(sig)
            if sim >= threshold:
                results.append((doc_id, sim))
        results.sort(key=lambda x: -x[1])
        return results


def _tokenize(text: str, ngram_size: int = 3) -> list[str]:
    """Produce character n-gram shingles from normalized text."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    normalized = re.sub(r"[^\w\s]", "", normalized)
    if len(normalized) < ngram_size:
        return [normalized] if normalized else []
    return [normalized[i : i + ngram_size] for i in range(len(normalized) - ngram_size + 1)]


def compute_minhash(text: str, num_perm: int = 128, ngram_size: int = 3) -> MinHash:
    """Compute MinHash signature for a text."""
    mh = MinHash(num_perm=num_perm)
    for token in _tokenize(text, ngram_size):
        mh.update(token)
    return mh


class Deduplicator:
    """MinHash-LSH deduplication for legal documents."""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        num_bands: int = 16,
    ) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = LSHIndex(num_perm=num_perm, num_bands=num_bands)
        self._doc_count = 0

    def add_document(self, doc_id: str, text: str) -> bool:
        """
        Add document to the index. Returns True if the document
        is a near-duplicate of an existing one.
        """
        mh = compute_minhash(text, num_perm=self.num_perm)
        duplicates = self.lsh.query_with_scores(mh, threshold=self.threshold)

        is_dup = len(duplicates) > 0
        if not is_dup:
            self.lsh.insert(doc_id, mh)
            self._doc_count += 1

        return is_dup

    def is_duplicate(self, text: str, known_hashes: Optional[Set[str]] = None) -> bool:
        """Check if text is a near-duplicate of any indexed document."""
        mh = compute_minhash(text, num_perm=self.num_perm)
        duplicates = self.lsh.query_with_scores(mh, threshold=self.threshold)
        return len(duplicates) > 0

    def get_similar_ids(self, text: str) -> list[str]:
        """Return IDs of similar documents above threshold."""
        mh = compute_minhash(text, num_perm=self.num_perm)
        results = self.lsh.query_with_scores(mh, threshold=self.threshold)
        return [doc_id for doc_id, _ in results]

    def get_similar_with_scores(self, text: str) -> list[tuple[str, float]]:
        """Return IDs and Jaccard similarity scores of similar documents."""
        mh = compute_minhash(text, num_perm=self.num_perm)
        return self.lsh.query_with_scores(mh, threshold=self.threshold)

    @property
    def document_count(self) -> int:
        return self._doc_count
