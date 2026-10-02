from enum import IntEnum, StrEnum


class RouteIntent(IntEnum):
    """Classification of query operational complexity / risk."""
    SIMPLE = 0
    CRITICAL = 1

    @property
    def label(self) -> str:
        return "Simples" if self == RouteIntent.SIMPLE else "Crítico"

    @property
    def badge_label(self) -> str:
        return "🟢 Simples / Econômico" if self == RouteIntent.SIMPLE else "🔴 Crítico / Alta Capacidade"


class RoutingStage(StrEnum):
    """The mechanism/stage that produced the routing decision."""
    FAST_PATH_HNSW = "Fast-Path HNSW"
    GATED_SLM = "Gated SLM"
    FAIL_SAFE = "Fallback: Sem logprobs (Fail-safe)"
