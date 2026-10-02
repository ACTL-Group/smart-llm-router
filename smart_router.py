import atexit
from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import time
from dotenv import load_dotenv
# noinspection PyPackageRequirements
import faiss
import numpy as np
from openai import OpenAI
import requests

load_dotenv()

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080").rstrip("/")
if SERVER_URL.endswith("/v1"):
    BASE_API_URL = SERVER_URL[:-3]
    OPENAI_BASE_URL = SERVER_URL
else:
    BASE_API_URL = SERVER_URL
    OPENAI_BASE_URL = f"{SERVER_URL}/v1"

API_KEY = os.getenv("API_KEY", "local-no-key")
MODEL_EMBED = os.getenv("MODEL_EMBED", "text-embedding")
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "bonsai-27b")
MODEL_EXPENSIVE = os.getenv("MODEL_EXPENSIVE", "qwen-27b")

MARGIN_THRESHOLD = float(os.getenv("ROUTER_MARGIN_THRESHOLD", "0.12"))
LOGPROB_THRESHOLD = float(os.getenv("ROUTER_LOGPROB_THRESHOLD", "-0.80"))

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=API_KEY)


class DynamicVRAMManager:

    def __init__(self, base_url: str, container_name: str = "dynamic-llama-server"):
        self.base_url = base_url
        self.container_name = container_name
        self.current_model: str | None = None

    def purge_all(self):
        for ep in ["/backend/shutdown", "/backend/stop", "/models/unload"]:
            try:
                requests.post(
                    f"{self.base_url}{ep}",
                    json={"all": True},
                    timeout=3,
                )
            except requests.RequestException:
                pass

        try:
            cmd = f"docker exec {self.container_name} pkill -f 'llama-cpp' || true"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        self.current_model = None
        time.sleep(1.0)

    def force_purge_model(self, model_name: str):
        for ep in ["/backend/shutdown", "/backend/stop", "/models/unload"]:
            try:
                requests.post(
                    f"{self.base_url}{ep}",
                    json={"backend": model_name, "model": model_name, "id": model_name},
                    timeout=3,
                )
            except requests.RequestException:
                pass

        try:
            cmd = f"docker exec {self.container_name} pkill -f '{model_name}' || true"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        time.sleep(1.2)

    def switch_to(self, model_name: str):
        if self.current_model == model_name:
            return

        if self.current_model is not None:
            print(f"{DIM}[VRAM] Desalocando {self.current_model} da GPU...{RESET}")
            self.force_purge_model(self.current_model)

        print(f"{DIM}[VRAM] Carregando {model_name} na GPU...{RESET}")
        self.current_model = model_name

    @contextmanager
    def session(self, model_name: str):
        self.switch_to(model_name)
        yield


vram = DynamicVRAMManager(BASE_API_URL)

# Limpeza no start
vram.purge_all()


def cleanup():
    vram.purge_all()


atexit.register(cleanup)


def handle_sigint(sig, frame):
    print(f"\n{YELLOW}Encerrando aplicação e limpando VRAM...{RESET}")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

CALIBRATION_SET = [
    ("Qual o horário de funcionamento das agências?", 0),
    ("Como faço para consultar meu saldo e extrato?", 0),
    ("Onde vejo o código de rastreamento do pedido?", 0),
    ("Vocês aceitam transferência via Pix para pagamento?", 0),
    ("Consta uma compra internacional no cartão que eu não fiz.", 1),
    ("Minha conta foi bloqueada por suspeita de fraude.", 1),
    ("Cobrança duplicada no aplicativo e suspensão de serviço.", 1),
    ("Caí em um golpe do Pix e transferi o valor por engano.", 1),
]


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL_EMBED, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def init_hnsw_index():
    print(f"{DIM}[INIT] Construindo índice semântico HNSW com {len(CALIBRATION_SET)} amostras...{RESET}")
    vectors_list = []
    labels_list = []

    for text, label in CALIBRATION_SET:
        vec = get_embedding(text)
        vectors_list.append(vec)
        labels_list.append(label)

    vectors = np.array(vectors_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 32
    index.add(vectors)
    print(f"{GREEN}[INIT] Índice pronto para inferência.{RESET}\n")
    return index, labels


HNSW_INDEX, TRAIN_LABELS = init_hnsw_index()


def evaluate_slm_uncertainty(query: str) -> int:
    with vram.session(MODEL_CHEAP):
        resp = client.chat.completions.create(
            model=MODEL_CHEAP,
            messages=[
                {
                    "role": "system",
                    "content": "Classifique a intenção em [SIMPLES] ou [CRITICO]. Responda apenas a tag.",
                },
                {"role": "user", "content": query},
            ],
            max_tokens=3,
            temperature=0.0,
            logprobs=True,
        )

    choice = resp.choices[0]
    token_logprobs = [
        item.logprob
        for item in (choice.logprobs.content or [])
        if item.logprob is not None
    ]

    if not token_logprobs or "[CRITICO]" in (choice.message.content or ""):
        return 1

    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return 1 if avg_logprob < LOGPROB_THRESHOLD else 0


def route_query(query: str) -> tuple[int, str]:
    q_vec = get_embedding(query)

    dists, idxs = HNSW_INDEX.search(np.expand_dims(q_vec, axis=0), k=3)
    labels = TRAIN_LABELS[idxs[0]]
    weights = dists[0]

    score_0 = float(np.sum(weights[labels == 0]))
    score_1 = float(np.sum(weights[labels == 1]))
    margin = abs(score_0 - score_1) / (max(score_0, score_1) + 1e-6)

    if margin >= MARGIN_THRESHOLD:
        target = 1 if score_1 >= score_0 else 0
        return target, f"Fast-Path HNSW (Margem: {margin:.2f})"

    target = evaluate_slm_uncertainty(query)
    return target, f"Gated Epistemic Fallback (Margem: {margin:.2f})"


def render_message(role: str, content: str, meta: str = ""):
    divider = "─" * 70
    if role == "user":
        header = f"{CYAN}{BOLD}🧑 Você{RESET}"
        text_color = CYAN
    elif role == "system":
        header = f"{YELLOW}{BOLD}⚙️  Sistema (Prompt de Contexto){RESET}"
        text_color = DIM
    else:
        model_badge = f"{MAGENTA}[{meta}]{RESET}" if meta else ""
        header = f"{GREEN}{BOLD}🤖 Assistente {model_badge}{RESET}"
        text_color = RESET

    print(f"\n{header}")
    print(f"{DIM}{divider}{RESET}")
    print(f"{text_color}{content.strip()}{RESET}")
    print(f"{DIM}{divider}{RESET}")


def dispatch(query: str):
    # 1. Mensagem do usuário
    render_message("user", query)

    # 2. Roteamento
    target, path_info = route_query(query)
    model_name = MODEL_CHEAP if target == 0 else MODEL_EXPENSIVE
    badge_label = "🟢 Simples / Econômico" if target == 0 else "🔴 Crítico / Especializado"

    print(f"{BLUE}↳ Decisão de Roteamento:{RESET} {badge_label} | {DIM}{path_info}{RESET}")

    if target == 0:
        system_prompt = "Você é um assistente de suporte operacional para dúvidas simples."
        render_message("system", system_prompt)

        with vram.session(MODEL_CHEAP):
            resp = client.chat.completions.create(
                model=MODEL_CHEAP,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
            )
        render_message("assistant", resp.choices[0].message.content or "", meta=MODEL_CHEAP)

    else:
        system_prompt = "Atendimento especializado em segurança, fraudes e ocorrências críticas."
        render_message("system", system_prompt)

        with vram.session(MODEL_EXPENSIVE):
            resp = client.chat.completions.create(
                model=MODEL_EXPENSIVE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
            )
        render_message("assistant", resp.choices[0].message.content or "", meta=MODEL_EXPENSIVE)


if __name__ == "__main__":
    test_queries = [
        "A agência abre no sábado para atendimento?",
        "Transferiram R$ 5.000 da minha conta por Pix e não fui eu.",
    ]

    print(f"{BOLD}=== Executando Queries de Teste ==={RESET}")
    for q in test_queries:
        dispatch(q)

    # Modo Interativo estilo chat
    print(f"\n{BOLD}=== Modo Chat Ativo (Digite 'sair' para encerrar) ==={RESET}")
    while True:
        try:
            user_input = input(f"\n{CYAN}{BOLD}Mensagem > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                break
            dispatch(user_input)
        except (KeyboardInterrupt, EOFError):
            break

    print(f"\n{YELLOW}Saindo...{RESET}")