import unittest
from unittest.mock import MagicMock
import numpy as np
from smart_router.config.settings import Settings
from smart_router.domain.enums import RouteIntent, RoutingStage
from smart_router.domain.models import CalibrationItem, NeighborMatch
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex
from smart_router.application.services.router_service import SmartRouterService


class TestRouterService(unittest.TestCase):

    def setUp(self):
        self.settings = Settings(
            router_margin_threshold=0.12,
            router_logprob_threshold=-0.80,
            model_cheap="bonsai-27b",
            model_expensive="qwen-27b",
        )
        self.mock_llm = MagicMock(spec=ILLMClient)
        self.mock_llm.get_embedding.return_value = np.array([1.0, 0.0], dtype=np.float32)
        self.mock_vector_index = MagicMock(spec=HNSWVectorIndex)
        self.mock_vector_index.raw_index = MagicMock()
        self.mock_vector_index.dim = 2

        self.router = SmartRouterService(
            settings=self.settings,
            llm_client=self.mock_llm,
            vector_index=self.mock_vector_index,
            calibration_items=[
                CalibrationItem("Simple query", RouteIntent.SIMPLE),
                CalibrationItem("Critical query", RouteIntent.CRITICAL),
            ],
        )

    def test_routing_fast_path_simple(self):
        self.mock_vector_index.get_neighbor_matches.return_value = [
            NeighborMatch(1, 0.95, RouteIntent.SIMPLE, "Simple query"),
        ]
        decision = self.router.route("Qual o horário?")
        self.assertEqual(decision.intent, RouteIntent.SIMPLE)
        self.assertEqual(decision.target_model, "bonsai-27b")
        self.assertEqual(decision.stage, RoutingStage.FAST_PATH_HNSW)
        self.assertIn("simples", decision.system_prompt.lower())

    def test_routing_fast_path_critical(self):
        self.mock_vector_index.get_neighbor_matches.return_value = [
            NeighborMatch(1, 0.95, RouteIntent.CRITICAL, "Critical query"),
        ]
        decision = self.router.route("Fraude no cartão")
        self.assertEqual(decision.intent, RouteIntent.CRITICAL)
        self.assertEqual(decision.target_model, "qwen-27b")
        self.assertEqual(decision.stage, RoutingStage.FAST_PATH_HNSW)
        self.assertIn("segurança", decision.system_prompt.lower())

    def test_routing_slm_fallback(self):
        # Ambiguous FastPath
        self.mock_vector_index.get_neighbor_matches.return_value = [
            NeighborMatch(1, 0.91, RouteIntent.CRITICAL, "Critical query"),
            NeighborMatch(2, 0.90, RouteIntent.SIMPLE, "Simple query"),
        ]
        # SLM says [SIMPLES] with high confidence
        self.mock_llm.evaluate_slm_logprobs.return_value = ("[SIMPLES]", [-0.10, -0.05])

        decision = self.router.route("Pergunta ambígua")
        self.assertEqual(decision.intent, RouteIntent.SIMPLE)
        self.assertEqual(decision.target_model, "bonsai-27b")
        self.assertEqual(decision.stage, RoutingStage.GATED_SLM)
        self.assertIsNotNone(decision.avg_logprob)


if __name__ == "__main__":
    unittest.main()
