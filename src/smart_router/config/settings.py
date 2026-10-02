from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application configuration parameters loaded from environment variables."""
    server_url: str = "http://localhost:8080"
    base_api_url: str = "http://localhost:8080"
    openai_base_url: str = "http://localhost:8080/v1"
    api_key: str = "local-no-key"
    model_embed: str = "text-embedding"
    model_cheap: str = "bonsai-27b"
    model_expensive: str = "qwen-27b"
    router_margin_threshold: float = 0.12
    router_logprob_threshold: float = -0.80
    container_name: str = "dynamic-llama-server"
    hnsw_m: int = 16
    hnsw_ef_search: int = 32
    k_neighbors: int = 3
    temperature_simple: float = 0.2
    temperature_critical: float = 0.3

    @classmethod
    def from_env(cls) -> "Settings":
        server_url = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080").rstrip("/")
        if server_url.endswith("/v1"):
            base_api_url = server_url[:-3]
            openai_base_url = server_url
        else:
            base_api_url = server_url
            openai_base_url = f"{server_url}/v1"

        return cls(
            server_url=server_url,
            base_api_url=base_api_url,
            openai_base_url=openai_base_url,
            api_key=os.getenv("API_KEY", "local-no-key"),
            model_embed=os.getenv("MODEL_EMBED", "text-embedding"),
            model_cheap=os.getenv("MODEL_CHEAP", "bonsai-27b"),
            model_expensive=os.getenv("MODEL_EXPENSIVE", "qwen-27b"),
            router_margin_threshold=float(os.getenv("ROUTER_MARGIN_THRESHOLD", "0.12")),
            router_logprob_threshold=float(os.getenv("ROUTER_LOGPROB_THRESHOLD", "-0.80")),
            container_name=os.getenv("CONTAINER_NAME", "dynamic-llama-server"),
            hnsw_m=int(os.getenv("HNSW_M", "16")),
            hnsw_ef_search=int(os.getenv("HNSW_EF_SEARCH", "32")),
            k_neighbors=int(os.getenv("ROUTER_K_NEIGHBORS", "3")),
            temperature_simple=float(os.getenv("TEMP_SIMPLE", "0.2")),
            temperature_critical=float(os.getenv("TEMP_CRITICAL", "0.3")),
        )
