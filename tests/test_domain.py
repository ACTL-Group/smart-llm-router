import unittest
from smart_router.domain.enums import RouteIntent, RoutingStage
from smart_router.domain.models import (
    CalibrationItem,
    NeighborMatch,
    FastPathResult,
    SLMEvaluationResult,
    RoutingDecision,
)
from smart_router.domain.constants import DEFAULT_CALIBRATION_SET


class TestDomainModels(unittest.TestCase):

    def test_route_intent_enum(self):
        self.assertEqual(int(RouteIntent.SIMPLE), 0)
        self.assertEqual(int(RouteIntent.CRITICAL), 1)
        self.assertEqual(RouteIntent.SIMPLE.label, "Simples")
        self.assertEqual(RouteIntent.CRITICAL.label, "Crítico")
        self.assertIn("Econômico", RouteIntent.SIMPLE.badge_label)
        self.assertIn("Alta Capacidade", RouteIntent.CRITICAL.badge_label)

    def test_calibration_item(self):
        item = CalibrationItem.from_tuple(("Dúvida teste", 0))
        self.assertEqual(item.text, "Dúvida teste")
        self.assertEqual(item.intent, RouteIntent.SIMPLE)

        self.assertGreater(len(DEFAULT_CALIBRATION_SET), 0)
        for cal in DEFAULT_CALIBRATION_SET:
            self.assertIsInstance(cal.intent, RouteIntent)
            self.assertIsInstance(cal.text, str)

    def test_routing_decision(self):
        decision = RoutingDecision(
            intent=RouteIntent.CRITICAL,
            target_model="qwen-27b",
            system_prompt="Security Prompt",
            temperature=0.3,
            badge_label=RouteIntent.CRITICAL.badge_label,
            source_info="Fast-Path HNSW (Margem: 0.50)",
            stage=RoutingStage.FAST_PATH_HNSW,
            margin=0.50,
        )
        self.assertEqual(decision.intent, RouteIntent.CRITICAL)
        self.assertEqual(decision.target_model, "qwen-27b")
        self.assertEqual(decision.margin, 0.50)
        self.assertIsNone(decision.avg_logprob)


if __name__ == "__main__":
    unittest.main()
