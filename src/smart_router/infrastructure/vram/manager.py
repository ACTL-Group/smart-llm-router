from abc import ABC, abstractmethod
from contextlib import contextmanager
import subprocess
import time
from typing import Generator
import requests


class IVRAMManager(ABC):
    """Interface for managing GPU VRAM and local inference model unloading."""

    @abstractmethod
    def purge_all(self) -> None:
        """Purge all models and orphan processes from VRAM."""
        pass

    @abstractmethod
    def force_purge_model(self, model_name: str) -> None:
        """Purge a specific model from VRAM."""
        pass

    @abstractmethod
    def switch_to(self, model_name: str) -> None:
        """Switch active model in VRAM, purging previous model if needed."""
        pass

    @abstractmethod
    @contextmanager
    def session(self, model_name: str) -> Generator[None, None, None]:
        """Context manager to ensure model is active during a block of work."""
        pass


class DynamicVRAMManager(IVRAMManager):
    """Dynamic VRAM manager interacting with LocalAI API and Docker container."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        container_name: str = "dynamic-llama-server",
        request_timeout: float = 3.0,
        purge_settle_time: float = 1.0,
        switch_settle_time: float = 1.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.container_name = container_name
        self.request_timeout = request_timeout
        self.purge_settle_time = purge_settle_time
        self.switch_settle_time = switch_settle_time
        self.current_model: str | None = None

    def purge_all(self) -> None:
        endpoints = ["/backend/shutdown", "/backend/stop", "/models/unload"]
        for ep in endpoints:
            try:
                requests.post(
                    f"{self.base_url}{ep}",
                    json={"all": True},
                    timeout=self.request_timeout,
                )
            except requests.RequestException:
                pass

        try:
            cmd = f"docker exec {self.container_name} pkill -f 'llama-cpp' || true"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        self.current_model = None
        if self.purge_settle_time > 0:
            time.sleep(self.purge_settle_time)

    def force_purge_model(self, model_name: str) -> None:
        endpoints = ["/backend/shutdown", "/backend/stop", "/models/unload"]
        for ep in endpoints:
            try:
                requests.post(
                    f"{self.base_url}{ep}",
                    json={"backend": model_name, "model": model_name, "id": model_name},
                    timeout=self.request_timeout,
                )
            except requests.RequestException:
                pass

        try:
            cmd = f"docker exec {self.container_name} pkill -f '{model_name}' || true"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        if self.switch_settle_time > 0:
            time.sleep(self.switch_settle_time)

    def switch_to(self, model_name: str) -> None:
        if self.current_model == model_name:
            return

        if self.current_model is not None:
            self.force_purge_model(self.current_model)

        self.current_model = model_name

    @contextmanager
    def session(self, model_name: str) -> Generator[None, None, None]:
        self.switch_to(model_name)
        yield
