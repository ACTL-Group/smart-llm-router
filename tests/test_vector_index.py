import unittest
import numpy as np
from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import CalibrationItem
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex


class TestHNSWVectorIndex(unittest.TestCase):

    def setUp(self):
        self.index = HNSWVectorIndex(dim=4, m=16, ef_search=32)
        self.items = [
            CalibrationItem("Query Simple 1", RouteIntent.SIMPLE),
            CalibrationItem("Query Simple 2", RouteIntent.SIMPLE),
            CalibrationItem("Query Critical 1", RouteIntent.CRITICAL),
            CalibrationItem("Query Critical 2", RouteIntent.CRITICAL),
        ]
        # 4D unit vectors
        self.embeddings = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32),
        ]

    def test_build_and_search_simple(self):
        self.index.build_index(self.items, self.embeddings)
        self.assertIsNotNone(self.index.raw_index)
        self.assertEqual(len(self.index.labels), 4)

        # Query close to Simple
        q_simple = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        matches = self.index.get_neighbor_matches(q_simple, k=2)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].intent, RouteIntent.SIMPLE)
        self.assertAlmostEqual(matches[0].score, 1.0, places=4)

    def test_search_critical(self):
        self.index.build_index(self.items, self.embeddings)

        # Query close to Critical
        q_critical = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        matches = self.index.get_neighbor_matches(q_critical, k=2)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].intent, RouteIntent.CRITICAL)
        self.assertAlmostEqual(matches[0].score, 1.0, places=4)

    def test_search_before_build_raises_error(self):
        uninitialized_index = HNSWVectorIndex()
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            uninitialized_index.search(q, k=2)


if __name__ == "__main__":
    unittest.main()
