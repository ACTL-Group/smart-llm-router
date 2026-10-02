from abc import ABC, abstractmethod
from typing import Sequence
import faiss
import numpy as np
from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import CalibrationItem, NeighborMatch


class IVectorIndex(ABC):
    """Interface for approximate nearest neighbor vector indexing."""

    @abstractmethod
    def build_index(self, items: Sequence[CalibrationItem], embeddings: Sequence[np.ndarray]) -> None:
        """Build the index with reference vectors and labels."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors, returning (distances, indices)."""
        pass


class HNSWVectorIndex(IVectorIndex):
    """FAISS HNSW Vector Index implementation with inner product / cosine distance."""

    def __init__(self, dim: int = 768, m: int = 16, ef_search: int = 32):
        self.dim = dim
        self.m = m
        self.ef_search = ef_search
        self._index: faiss.IndexHNSWFlat | None = None
        self._labels: np.ndarray | None = None
        self._reference_items: list[CalibrationItem] = []

    @property
    def raw_index(self) -> faiss.IndexHNSWFlat | None:
        return self._index

    @property
    def labels(self) -> np.ndarray | None:
        return self._labels

    @property
    def reference_items(self) -> list[CalibrationItem]:
        return self._reference_items

    def build_index(self, items: Sequence[CalibrationItem], embeddings: Sequence[np.ndarray]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        labels = np.array([int(item.intent) for item in items], dtype=np.int32)
        dim = vectors.shape[1]
        self.dim = dim

        index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.ef_search
        index.add(vectors)

        self._index = index
        self._labels = labels
        self._reference_items = list(items)

    def search(self, query_vector: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._labels is None:
            raise RuntimeError("HNSW Vector Index has not been initialized. Build index first.")

        query_expanded = np.expand_dims(query_vector.astype(np.float32), axis=0)
        distances, indices = self._index.search(query_expanded, k)
        return distances[0], indices[0]

    def get_neighbor_matches(self, query_vector: np.ndarray, k: int = 3) -> list[NeighborMatch]:
        distances, indices = self.search(query_vector, k=k)
        matches: list[NeighborMatch] = []
        for rank, (idx, score) in enumerate(zip(indices, distances), start=1):
            lbl = self._labels[idx]
            ref_text = self._reference_items[idx].text if idx < len(self._reference_items) else ""
            matches.append(
                NeighborMatch(
                    rank=rank,
                    score=float(score),
                    intent=RouteIntent(lbl),
                    reference_text=ref_text,
                )
            )
        return matches
