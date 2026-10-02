import time
from smart_router.config.settings import Settings
from smart_router.domain.constants import SYSTEM_PROMPT_SLM_CLASSIFICATION
from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import SLMEvaluationResult
from smart_router.infrastructure.llm.client import ILLMClient
from smart_router.presentation.console import ConsoleRenderer


class SLMUncertaintyEvaluator:
    """Evaluates query complexity uncertainty using small language model (SLM) logprobs."""

    def __init__(
        self,
        settings: Settings,
        llm_client: ILLMClient,
        renderer: ConsoleRenderer | None = None,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.renderer = renderer

    def evaluate(self, query: str) -> SLMEvaluationResult:
        if self.renderer:
            self.renderer.log_step(
                "FALLBACK EPISTÊMICO (SLM)",
                "Margem de embeddings ambígua. Avaliando incerteza probabilística com SLM.",
            )

        t0 = time.time()
        raw_text, token_logprobs = self.llm_client.evaluate_slm_logprobs(
            query=query,
            system_prompt=SYSTEM_PROMPT_SLM_CLASSIFICATION,
            model=self.settings.model_cheap,
            max_tokens=3,
        )
        elapsed = time.time() - t0

        if not token_logprobs:
            if self.renderer:
                self.renderer.log_sub(
                    f"Sem logprobs retornados. Forçando rota de segurança para Crítico ({elapsed:.2f}s)."
                )
                self.renderer.log_end()

            return SLMEvaluationResult(
                intent=RouteIntent.CRITICAL,
                raw_text=raw_text,
                avg_logprob=None,
                reason="Fallback: Sem logprobs (Fail-safe)",
                source_info="Fallback: Sem logprobs (Fail-safe)",
                elapsed_seconds=elapsed,
            )

        avg_logprob = sum(token_logprobs) / len(token_logprobs)

        if self.renderer:
            self.renderer.log_sub(f"Classificação gerada: '{raw_text}'")
            self.renderer.log_sub(
                f"Logprob médio: {avg_logprob:.4f} (Limiar de corte: {self.settings.router_logprob_threshold})"
            )

        if "[CRITICO]" in raw_text or avg_logprob < self.settings.router_logprob_threshold:
            intent = RouteIntent.CRITICAL
            motivo = "Incerteza alta do SLM ou tag [CRITICO]"
        else:
            intent = RouteIntent.SIMPLE
            motivo = "Confiança alta do SLM em [SIMPLES]"

        if self.renderer:
            self.renderer.log_end(f"Resultado do SLM: {motivo} em {elapsed:.2f}s")

        return SLMEvaluationResult(
            intent=intent,
            raw_text=raw_text,
            avg_logprob=avg_logprob,
            reason=motivo,
            source_info=f"Gated SLM (AvgLogprob: {avg_logprob:.2f})",
            elapsed_seconds=elapsed,
        )
