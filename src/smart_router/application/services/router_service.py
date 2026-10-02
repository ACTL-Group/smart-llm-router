from typing import Sequence
import time
from smart_router.config.settings import Settings
from smart_router.domain.constants import (
    DEFAULT_CALIBRATION_SET,
    SYSTEM_PROMPT_CRITICAL,
    SYSTEM_PROMPT_SIMPLE,
)
from smart_router.domain.enums import RouteIntent, RoutingStage
from smart_router.domain.models import CalibrationItem, RoutingDecision
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex
from smart_router.application.services.fast_path_router import SemanticFastPathRouter
from smart_router.application.services.slm_evaluator import SLMUncertaintyEvaluator
from smart_router.presentation.console import ConsoleRenderer


class SmartRouterService:
    """Orchestrates hybrid semantic HNSW routing with SLM epistemic fallback."""

    def __init__(
        self,
        settings: Settings,
        llm_client: ILLMClient,
        vector_index: HNSWVectorIndex,
        calibration_items: Sequence[CalibrationItem] | None = None,
        renderer: ConsoleRenderer | None = None,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.vector_index = vector_index
        self.calibration_items = list(calibration_items or DEFAULT_CALIBRATION_SET)
        self.renderer = renderer

        self.fast_path_router = SemanticFastPathRouter(
            settings=settings,
            llm_client=llm_client,
            vector_index=vector_index,
            renderer=renderer,
        )
        self.slm_evaluator = SLMUncertaintyEvaluator(
            settings=settings,
            llm_client=llm_client,
            renderer=renderer,
        )
        self._is_initialized = False

    def initialize_index(self) -> None:
        """Calibrate vector space and populate HNSW index with calibration set."""
        if self._is_initialized and self.vector_index.raw_index is not None:
            return

        if self.renderer:
            self.renderer.log_step(
                "INDEXAÇÃO HNSW",
                f"Calibrando espaço vetorial com {len(self.calibration_items)} intenções de referência",
            )

        t0 = time.time()
        embeddings = [
            self.llm_client.get_embedding(item.text, model=self.settings.model_embed)
            for item in self.calibration_items
        ]

        self.vector_index.build_index(self.calibration_items, embeddings)
        elapsed = time.time() - t0

        if self.renderer:
            self.renderer.log_sub(
                f"Dimensão vetorial: {self.vector_index.dim}D | Métrica: Cosine Distance (via Inner Product)"
            )
            self.renderer.log_end(f"Grafo HNSW pronto em {elapsed:.2f}s")

        self._is_initialized = True

    def route(self, query: str) -> RoutingDecision:
        """Process user query and return definitive routing decision."""
        if not self._is_initialized or self.vector_index.raw_index is None:
            self.initialize_index()

        fast_path = self.fast_path_router.evaluate(query)

        if fast_path.is_confident:
            intent = fast_path.intent
            stage = RoutingStage.FAST_PATH_HNSW
            source_info = fast_path.source_info
            margin = fast_path.margin
            avg_logprob = None
        else:
            slm_result = self.slm_evaluator.evaluate(query)
            intent = slm_result.intent
            stage = (
                RoutingStage.FAIL_SAFE
                if slm_result.avg_logprob is None
                else RoutingStage.GATED_SLM
            )
            source_info = slm_result.source_info
            margin = fast_path.margin
            avg_logprob = slm_result.avg_logprob

        target_model = (
            self.settings.model_cheap
            if intent == RouteIntent.SIMPLE
            else self.settings.model_expensive
        )
        system_prompt = (
            SYSTEM_PROMPT_SIMPLE
            if intent == RouteIntent.SIMPLE
            else SYSTEM_PROMPT_CRITICAL
        )
        temperature = (
            self.settings.temperature_simple
            if intent == RouteIntent.SIMPLE
            else self.settings.temperature_critical
        )

        return RoutingDecision(
            intent=intent,
            target_model=target_model,
            system_prompt=system_prompt,
            temperature=temperature,
            badge_label=intent.badge_label,
            source_info=source_info,
            stage=stage,
            margin=margin,
            avg_logprob=avg_logprob,
        )
