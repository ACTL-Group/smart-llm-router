from contextlib import contextmanager
import os
import time
from dotenv import load_dotenv
# noinspection PyPackageRequirements
import faiss
import numpy as np
from openai import OpenAI
import requests

load_dotenv()

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

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.current_model: str | None = None

    def switch_to(self, model_name: str):
        if self.current_model == model_name:
            return

        if self.current_model is not None:
            try:
                requests.post(
                    f"{self.base_url}/models/unload",
                    json={"model": self.current_model},
                    timeout=15,
                )
            except requests.RequestException:
                pass
            time.sleep(1.0)

        self.current_model = model_name

    @contextmanager
    def session(self, model_name: str):
        self.switch_to(model_name)
        yield


vram = DynamicVRAMManager(BASE_API_URL)

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
        return target, f"Fast-Path HNSW (Margin: {margin:.2f})"

    target = evaluate_slm_uncertainty(query)
    return target, f"Gated Epistemic Fallback (Margin: {margin:.2f})"


def dispatch(query: str) -> str:
    target, path_info = route_query(query)

    if target == 0:
        with vram.session(MODEL_CHEAP):
            resp = client.chat.completions.create(
                model=MODEL_CHEAP,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um assistente de suporte operacional para dúvidas simples.",
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
            )
        return (
            f"[Rota: Barato Local ({MODEL_CHEAP}) | Caminho: {path_info}]\n"
            f"{resp.choices[0].message.content}"
        )

    with vram.session(MODEL_EXPENSIVE):
        resp = client.chat.completions.create(
            model=MODEL_EXPENSIVE,
            messages=[
                {
                    "role": "system",
                    "content": "Atendimento especializado em segurança, fraudes e ocorrências críticas.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
    return (
        f"[Rota: Caro Local ({MODEL_EXPENSIVE}) | Caminho: {path_info}]\n"
        f"{resp.choices[0].message.content}"
    )


if __name__ == "__main__":
    test_queries = [
        "A agência abre no sábado para atendimento?",
        "Transferiram R$ 5.000 da minha conta por Pix e não fui eu.",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        print(dispatch(q))
        print("=" * 60)