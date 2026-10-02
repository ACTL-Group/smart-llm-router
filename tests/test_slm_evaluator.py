import unittest
from unittest.mock import MagicMock
from smart_router.config.settings import Settings
from smart_router.domain.enums import RouteIntent
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.application.services.slm_evaluator import SLMUncertaintyEvaluator


class TestSLMEvaluator(unittest.TestCase):

    def setUp(self):
        self.settings = Settings(router_logprob_threshold=-0.80)
        self.mock_llm = MagicMock(spec=ILLMClient)
        self.evaluator = SLMUncertaintyEvaluator(
            settings=self.settings,
            llm_client=self.mock_llm,
        )

    def test_slm_confident_simple(self):
        self.mock_llm.evaluate_slm_logprobs.return_value = ("[SIMPLES]", [-0.10, -0.05])
        result = self.evaluator.evaluate("Como vejo meu saldo?")
        self.assertEqual(result.intent, RouteIntent.SIMPLE)
        self.assertAlmostEqual(result.avg_logprob, -0.075)

    def test_slm_tag_critico(self):
        self.mock_llm.evaluate_slm_logprobs.return_value = ("[CRITICO]", [-0.10, -0.05])
        result = self.evaluator.evaluate("Bloquearam meu cartão")
        self.assertEqual(result.intent, RouteIntent.CRITICAL)

    def test_slm_uncertain_low_logprob(self):
        # Even with [SIMPLES] tag, low average logprob (< -0.80) should route to CRITICAL
        self.mock_llm.evaluate_slm_logprobs.return_value = ("[SIMPLES]", [-0.95, -1.20])
        result = self.evaluator.evaluate("Transação estranha")
        self.assertEqual(result.intent, RouteIntent.CRITICAL)
        self.assertIn("Incerteza alta", result.reason)

    def test_slm_failsafe_no_logprobs(self):
        self.mock_llm.evaluate_slm_logprobs.return_value = ("", [])
        result = self.evaluator.evaluate("Fallback query")
        self.assertEqual(result.intent, RouteIntent.CRITICAL)
        self.assertIsNone(result.avg_logprob)
        self.assertIn("Fail-safe", result.reason)


if __name__ == "__main__":
    unittest.main()
