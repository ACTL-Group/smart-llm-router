from smart_router.config.settings import Settings
from smart_router.domain.constants import BOLD, CYAN, RESET, YELLOW
from smart_router.infrastructure.llm.client import OpenAILLMClient
from smart_router.infrastructure.system.lifecycle import register_shutdown_handlers
from smart_router.infrastructure.vector_store.hnsw_index import HNSWVectorIndex
from smart_router.infrastructure.vram.manager import DynamicVRAMManager
from smart_router.application.services.router_service import SmartRouterService
from smart_router.application.services.chat_service import ChatService
from smart_router.presentation.console import ConsoleRenderer


def main() -> None:
    settings = Settings.from_env()
    renderer = ConsoleRenderer()

    # Initialize infrastructure
    vram = DynamicVRAMManager(
        base_url=settings.base_api_url,
        container_name=settings.container_name,
    )
    register_shutdown_handlers(
        vram_manager=vram,
        on_sigint_message=renderer.render_sigint_message,
    )

    llm_client = OpenAILLMClient(settings=settings, vram_manager=vram)
    vector_index = HNSWVectorIndex(
        dim=768,
        m=settings.hnsw_m,
        ef_search=settings.hnsw_ef_search,
    )

    # Initialize application services
    router_service = SmartRouterService(
        settings=settings,
        llm_client=llm_client,
        vector_index=vector_index,
        renderer=renderer,
    )
    chat_service = ChatService(
        settings=settings,
        router_service=router_service,
        llm_client=llm_client,
        vram_manager=vram,
        renderer=renderer,
    )

    # Step 1: Environment initialization & initial VRAM wipe
    renderer.log_step("INICIALIZAÇÃO DO AMBIENTE", "Garantindo que a GPU comece vazia")
    vram.purge_all()
    renderer.log_end("VRAM zerada")

    # Step 2: Index initialization
    router_service.initialize_index()

    # Step 3: Run automated test queries
    test_queries = [
        "A agência abre no sábado para atendimento?",
        "Transferiram R$ 5.000 da minha conta por Pix e não fui eu.",
    ]

    print(f"{BOLD}=== INICIANDO BATERIA DE TESTES AUTOMATIZADOS ==={RESET}")
    for q in test_queries:
        chat_service.dispatch(q)
        print("=" * 80)

    # Step 4: Interactive terminal chat
    print(f"\n{BOLD}=== TERMINAL DE CHAT INTERATIVO (Digite 'sair' para encerrar) ==={RESET}")
    while True:
        try:
            user_input = input(f"\n{CYAN}{BOLD}Mensagem > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                break
            chat_service.dispatch(user_input)
        except (KeyboardInterrupt, EOFError):
            break

    print(f"\n{YELLOW}Desligando...{RESET}")


if __name__ == "__main__":
    main()
