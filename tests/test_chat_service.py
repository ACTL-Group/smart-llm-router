import unittest
from unittest.mock import MagicMock
from smart_router.config.settings import Settings
from smart_router.domain.enums import RouteIntent, RoutingStage
from smart_router.domain.models import RoutingDecision
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.infrastructure.vram.manager import IVRAMManager
from smart_router.application.services.router_service import SmartRouterService
from smart_router.application.services.chat_service import ChatService


class TestChatService(unittest.TestCase):

    def setUp(self):
        self.settings = Settings()
        self.mock_router = MagicMock(spec=SmartRouterService)
        self.mock_llm = MagicMock(spec=ILLMClient)
        self.mock_vram = MagicMock(spec=IVRAMManager)

        self.chat_service = ChatService(
            settings=self.settings,
            router_service=self.mock_router,
            llm_client=self.mock_llm,
            vram_manager=self.mock_vram,
        )

    def test_dispatch_simple_flow(self):
        decision = RoutingDecision(
            intent=RouteIntent.SIMPLE,
            target_model="bonsai-27b",
            system_prompt="Simple Prompt",
            temperature=0.2,
            badge_label=RouteIntent.SIMPLE.badge_label,
            source_info="Fast-Path HNSW (Margem: 0.50)",
            stage=RoutingStage.FAST_PATH_HNSW,
        )
        self.mock_router.route.return_value = decision
        self.mock_llm.generate_chat_completion.return_value = "Horário de 10h às 16h."

        response = self.chat_service.dispatch("Qual o horário?")
        self.assertEqual(response, "Horário de 10h às 16h.")
        self.mock_router.route.assert_called_once_with("Qual o horário?")
        self.mock_llm.generate_chat_completion.assert_called_once_with(
            messages=[
                {"role": "system", "content": "Simple Prompt"},
                {"role": "user", "content": "Qual o horário?"},
            ],
            model="bonsai-27b",
            temperature=0.2,
        )


if __name__ == "__main__":
    unittest.main()
