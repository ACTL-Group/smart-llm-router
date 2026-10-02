from smart_router.domain.constants import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RESET,
    YELLOW,
)
from smart_router.domain.models import RoutingDecision


class ConsoleRenderer:
    """Renders formatted console output with ANSI colors and headers."""

    def log_step(self, title: str, detail: str = "") -> None:
        print(f"\n{BLUE}{BOLD}┌── [ETAPA: {title}]{RESET}")
        if detail:
            print(f"{BLUE}│{RESET}  {detail}")

    def log_sub(self, text: str) -> None:
        print(f"{BLUE}│{RESET}  {text}")

    def log_end(self, summary: str = "") -> None:
        if summary:
            print(f"{BLUE}│{RESET}  {GREEN}✔ {summary}{RESET}")
        print(f"{BLUE}└──{RESET}")

    def render_chat_turn(self, role: str, content: str, meta: str = "") -> None:
        divider = "─" * 70
        if role == "user":
            header = f"{CYAN}{BOLD}🧑 Você{RESET}"
            text_color = CYAN
        elif role == "system":
            header = f"{YELLOW}{BOLD}⚙️  Prompt de Sistema Injetado{RESET}"
            text_color = DIM
        else:
            model_badge = f"{MAGENTA}[{meta}]{RESET}" if meta else ""
            header = f"{GREEN}{BOLD}🤖 Assistente {model_badge}{RESET}"
            text_color = RESET

        print(f"\n{header}")
        print(f"{DIM}{divider}{RESET}")
        print(f"{text_color}{content.strip()}{RESET}")
        print(f"{DIM}{divider}{RESET}\n")

    def render_decision(self, decision: RoutingDecision) -> None:
        print(f"\n{BOLD}⚡ DECISÃO FINAL DE ROTEAMENTO:{RESET} {decision.badge_label}")
        print(f"{DIM}Modelo selecionado: {decision.target_model} | Origem: {decision.source_info}{RESET}")

    def render_sigint_message(self) -> None:
        print(f"\n{YELLOW}Interrupção detectada. Ejetando modelos da GPU antes de sair...{RESET}")
