import unittest
from unittest.mock import MagicMock
import numpy as np
from smart_router.config.settings import Settings
from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import CalibrationItem, NeighborMatch
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex
from smart_router.application.services.fast_path_router import SemanticFastPathRouter


class TestFastPathRouter(unittest.TestCase):

    def setUp(self):
        self.settings = Settings(router_margin_threshold=0.12, k_neighbors=3)
        self.mock_llm = MagicMock(spec=ILLMClient)
        self.mock_llm.get_embedding.return_value = np.array([1.0, 0.0], dtype=np.float32)
        self.mock_vector_index = MagicMock(spec=HNSWVectorIndex)

        self.router = SemanticFastPathRouter(
            settings=self.settings,
            llm_client=self.mock_llm,
            vector_index=self.mock_vector_index,
        )

    def test_high_confidence_simple(self):
        # 3 neighbors matching SIMPLE with high scores
        self.mock_vector_index.get_neighbor_matches.return_value = [
            NeighborMatch(1, 0.95, RouteIntent.SIMPLE, "Simples 1"),
            NeighborMatch(2, 0.90, RouteIntent.SIMPLE, "Simples 2"),
            NeighborMatch(3, 0.85, RouteIntent.SIMPLE, "Simples 3"),
        ]

        result = self.router.evaluate("Consulta simples")
        self.assertTrue(result.is_confident)
        self.assertEqual(result.intent, RouteIntent.SIMPLE)
        self.assertGreaterEqual(result.margin, 0.12)

    def test_high_confidence_critical(self):
        # 3 neighbors matching CRITICAL with high scores
        self.mock_vector_index.get_neighbor_matches.return_value = [
            NeighborMatch(1, 0.95, RouteIntent.CRITICAL, "Crítico 1"),
            NeighborMatch(2, 0.90, RouteIntent.CRITICAL, "Crítico 2"),
            NeighborMatch(3, 0.85, RouteIntent.CRITICAL, "Crítico 3"),
        ]

        result = self.router.evaluate("Golpe no Pix")
        self.assertTrue(result.is_confident)
        self.assertEqual(result.intent, RouteIntent.CRITICAL)
        self.assertGreaterEqual(result.margin, 0.12)

    def test_low_confidence_ambiguous_margin(self):
        # Ambiguous matches: score_simple ~ 0.90, score_critical ~ 0.92 -> margin small
        self.mock_vector_index.get_neighbor_matches.return_value = [
            NeighborMatch(1, 0.92, RouteIntent.CRITICAL, "Crítico 1"),
            NeighborMatch(2, 0.90, RouteIntent.SIMPLE, "Simples 1"),
        ]

        result = self.router.evaluate("Dúvida ambígua")
        self.assertFalse(result.is_confident)
        self.assertLess(result.margin, 0.12)


if __name__ == "__main__":
    unittest.main()
