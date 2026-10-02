import time
from smart_router.config.settings import Settings
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.infrastructure.vram.manager import IVRAMManager
from smart_router.application.services.router_service import SmartRouterService
from smart_router.presentation.console import ConsoleRenderer


class ChatService:
    """Handles end-to-end user query execution, routing, inference, and display."""

    def __init__(
        self,
        settings: Settings,
        router_service: SmartRouterService,
        llm_client: ILLMClient,
        vram_manager: IVRAMManager | None = None,
        renderer: ConsoleRenderer | None = None,
    ):
        self.settings = settings
        self.router_service = router_service
        self.llm_client = llm_client
        self.vram_manager = vram_manager
        self.renderer = renderer

    def dispatch(self, query: str) -> str:
        """Route user query, perform LLM inference with dynamic VRAM management, and render output."""
        if self.renderer:
            self.renderer.render_chat_turn("user", query)

        decision = self.router_service.route(query)

        if self.renderer:
            self.renderer.render_decision(decision)
            self.renderer.render_chat_turn("system", decision.system_prompt)
            self.renderer.log_step(
                "GERAÇÃO DE RESPOSTA",
                f"Enviando contexto para inferência com '{decision.target_model}'",
            )

        t0 = time.time()
        messages = [
            {"role": "system", "content": decision.system_prompt},
            {"role": "user", "content": query},
        ]

        response_text = self.llm_client.generate_chat_completion(
            messages=messages,
            model=decision.target_model,
            temperature=decision.temperature,
        )
        elapsed = time.time() - t0

        if self.renderer:
            self.renderer.log_end(f"Resposta gerada em {elapsed:.2f}s")
            self.renderer.render_chat_turn("assistant", response_text, meta=decision.target_model)

        return response_text
