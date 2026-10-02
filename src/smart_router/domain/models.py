from dataclasses import dataclass, field
from smart_router.domain.enums import RouteIntent, RoutingStage


@dataclass(frozen=True)
class CalibrationItem:
    """Reference item used to calibrate the semantic vector index."""
    text: str
    intent: RouteIntent

    @classmethod
    def from_tuple(cls, item: tuple[str, int]) -> "CalibrationItem":
        return cls(text=item[0], intent=RouteIntent(item[1]))


@dataclass(frozen=True)
class NeighborMatch:
    """Nearest neighbor search match from vector index."""
    rank: int
    score: float
    intent: RouteIntent
    reference_text: str


@dataclass(frozen=True)
class FastPathResult:
    """Result of HNSW fast-path semantic classification."""
    is_confident: bool
    intent: RouteIntent
    margin: float
    score_simple: float
    score_critical: float
    neighbors: list[NeighborMatch] = field(default_factory=list)
    source_info: str = ""


@dataclass(frozen=True)
class SLMEvaluationResult:
    """Result of epistemic SLM logprob and tag evaluation."""
    intent: RouteIntent
    raw_text: str
    avg_logprob: float | None
    reason: str
    source_info: str
    elapsed_seconds: float


@dataclass(frozen=True)
class RoutingDecision:
    """Final routing decision determining target model and configuration."""
    intent: RouteIntent
    target_model: str
    system_prompt: str
    temperature: float
    badge_label: str
    source_info: str
    stage: RoutingStage
    margin: float | None = None
    avg_logprob: float | None = None
