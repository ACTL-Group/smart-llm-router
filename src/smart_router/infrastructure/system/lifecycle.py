import atexit
import signal
import sys
from typing import Callable
from smart_router.infrastructure.vram.manager import IVRAMManager


def register_shutdown_handlers(
    vram_manager: IVRAMManager,
    on_sigint_message: Callable[[], None] | None = None,
) -> None:
    """Register graceful shutdown handlers for process exit and OS signals."""

    def cleanup():
        vram_manager.purge_all()

    atexit.register(cleanup)

    def handle_signal(sig, frame):
        if on_sigint_message:
            on_sigint_message()
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
