import sys
from pathlib import Path

# Ensure 'src' is in sys.path and configure package path
_src_dir = Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Enable submodule imports when smart_router.py is loaded as top-level module
__path__ = [str(_src_dir / "smart_router")]

from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import RoutingDecision, CalibrationItem
from smart_router.config.settings import Settings
from smart_router.infrastructure.vram.manager import DynamicVRAMManager
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex
from smart_router.infrastructure.llm.client import OpenAILLMClient
from smart_router.application.services.router_service import SmartRouterService
from smart_router.application.services.chat_service import ChatService
from smart_router.presentation.console import ConsoleRenderer
from smart_router.presentation.cli import main

# Re-export key variables and functions for backwards compatibility
_settings = Settings.from_env()
_renderer = ConsoleRenderer()
_vram = DynamicVRAMManager(base_url=_settings.base_api_url, container_name=_settings.container_name)
_llm_client = OpenAILLMClient(settings=_settings, vram_manager=_vram)
_vector_index = HNSWVectorIndex(dim=768, m=_settings.hnsw_m, ef_search=_settings.hnsw_ef_search)
_router = SmartRouterService(
    settings=_settings,
    llm_client=_llm_client,
    vector_index=_vector_index,
    renderer=_renderer,
)
_chat_service = ChatService(
    settings=_settings,
    router_service=_router,
    llm_client=_llm_client,
    vram_manager=_vram,
    renderer=_renderer,
)

# Compatibility global references
vram = _vram
client = _llm_client.raw_client
MODEL_EMBED = _settings.model_embed
MODEL_CHEAP = _settings.model_cheap
MODEL_EXPENSIVE = _settings.model_expensive
MARGIN_THRESHOLD = _settings.router_margin_threshold
LOGPROB_THRESHOLD = _settings.router_logprob_threshold
CALIBRATION_SET = [(item.text, int(item.intent)) for item in _router.calibration_items]


def log_step(title: str, detail: str = ""):
    _renderer.log_step(title, detail)


def log_sub(text: str):
    _renderer.log_sub(text)


def log_end(summary: str = ""):
    _renderer.log_end(summary)


def get_embedding(text: str):
    return _llm_client.get_embedding(text)


def init_hnsw_index():
    _router.initialize_index()
    return _vector_index.raw_index, _vector_index.labels


def evaluate_slm_uncertainty(query: str):
    res = _router.slm_evaluator.evaluate(query)
    return int(res.intent), res.reason


def route_query(query: str):
    decision = _router.route(query)
    return int(decision.intent), decision.source_info


def render_chat_turn(role: str, content: str, meta: str = ""):
    _renderer.render_chat_turn(role, content, meta)


def dispatch(query: str):
    _chat_service.dispatch(query)


if __name__ == "__main__":
    main()
