import time
import numpy as np
from smart_router.config.settings import Settings
from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import FastPathResult
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex
from smart_router.presentation.console import ConsoleRenderer


class SemanticFastPathRouter:
    """Semantic vector search router using HNSW graph nearest neighbors."""

    def __init__(
        self,
        settings: Settings,
        llm_client: ILLMClient,
        vector_index: HNSWVectorIndex,
        renderer: ConsoleRenderer | None = None,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.vector_index = vector_index
        self.renderer = renderer

    def evaluate(self, query: str) -> FastPathResult:
        if self.renderer:
            self.renderer.log_step(
                "VETORIZAÇÃO E BUSCA SEMÂNTICA",
                f"Convertendo prompt para embedding via '{self.settings.model_embed}'",
            )

        t0 = time.time()
        q_vec = self.llm_client.get_embedding(query)
        if self.renderer:
            self.renderer.log_sub(f"Vetor gerado e normalizado em {time.time() - t0:.2f}s")
            self.renderer.log_sub(
                f"Consultando os k={self.settings.k_neighbors} vizinhos mais próximos no grafo HNSW..."
            )

        matches = self.vector_index.get_neighbor_matches(q_vec, k=self.settings.k_neighbors)

        if self.renderer:
            for match in matches:
                self.renderer.log_sub(
                    f"  #{match.rank} [Score: {match.score:.3f}] [{match.intent.label}] \"{match.reference_text}\""
                )

        scores = {RouteIntent.SIMPLE: 0.0, RouteIntent.CRITICAL: 0.0}
        for match in matches:
            scores[match.intent] += match.score

        score_0 = scores[RouteIntent.SIMPLE]
        score_1 = scores[RouteIntent.CRITICAL]
        margin = abs(score_0 - score_1) / (max(score_0, score_1) + 1e-6)

        if self.renderer:
            self.renderer.log_sub(
                f"Score Simples (0): {score_0:.3f} | Score Crítico (1): {score_1:.3f}"
            )
            self.renderer.log_sub(
                f"Margem de confiança calculada: {margin:.4f} (Threshold mínimo: {self.settings.router_margin_threshold})"
            )

        is_confident = margin >= self.settings.router_margin_threshold
        intent = RouteIntent.CRITICAL if score_1 >= score_0 else RouteIntent.SIMPLE

        source_info = f"Fast-Path HNSW (Margem: {margin:.2f})"

        if is_confident and self.renderer:
            self.renderer.log_end(
                f"Fast-Path HNSW determinou rota '{intent.label}' com margem {margin:.2f}"
            )
        elif not is_confident and self.renderer:
            self.renderer.log_sub("Margem abaixo do threshold. O classificador semântico está incerto.")
            self.renderer.log_end()

        return FastPathResult(
            is_confident=is_confident,
            intent=intent,
            margin=margin,
            score_simple=score_0,
            score_critical=score_1,
            neighbors=matches,
            source_info=source_info,
        )
