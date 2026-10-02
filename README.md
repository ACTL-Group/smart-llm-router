# Smart LLM Router 🚦🧠

> Roteador semântico híbrido e adaptativo para redução de custos com LLMs, combinando busca vetorial em grafos (HNSW) e inspeção de incerteza por logprobs no modelo local antes do transbordo para APIs de fronteira.

## 📌 Visão Geral

O envio indiscriminado de todas as mensagens de suporte ao cliente para APIs de ponta (como GPT-4o) impõe custos elevados de tokens e latência desnecessária para dúvidas frequentes. Por outro lado, confiar exclusivamente em modelos locais leves (Small Language Models - SLMs) expõe a operação ao risco de falhas em casos de fraude, vazamento de dados ou contestações financeiras.

O **Smart LLM Router** atua como uma camada intermediária de triagem inteligente logo na chegada da mensagem do usuário:

1. **Casos Canônicos e FAQ:** Resolvidos em menos de 3 ms via busca vetorial em grafos aproximados (HNSW) e despachados para o modelo local ou base interna a custo marginal próximo de zero.

2. **Casos Ambíguos (Zona Cinzenta):** Quando a margem semântica é estreita (`Δ_margin < τ`), o sistema executa uma micro-inferência de 3 tokens no SLM local para avaliar a entropia dos logprobs.

3. **Casos Críticos / Fraudes:** Identificada a urgência ou incerteza epistêmica, o chamado é encaminhado com prioridade defensiva para o modelo de fronteira em nuvem ou transbordo humano.

## 🏗️ Arquitetura de Decisão

```text
               [ Mensagem do Usuário ]
                          │
                          ▼
            [ Embedding Local via LM Studio ]
             (nomic-embed-text-v1.5 / L2 Norm)
                          │
                          ▼
            [ Busca Vetorial HNSW (FAISS) ]
                 k=3 vizinhos mais próximos
                          │
                 Δ_margin >= 0.12?
                ┌─────────┴─────────┐
               Sim                 Não (Ambiguidade Léxica)
                │                   │
                ▼                   ▼
        [ Decisão Direta ]   [ Inspeção por Incerteza (Logprobs) ]
          (HNSW k-NN)        (Micro-inferência: Llama 3 - 3 tokens)
                │                   │
                │            Incerteza Alta / Tag [CRITICO]?
                │                   ┌───────┴───────┐
                │                  Sim             Não
                │                   │               │
                ▼                   ▼               ▼
      ┌──────────────────────┐  ┌─────────────────────────────────┐
      │ Rota 0: Baixo Custo  │  │ Rota 1: Alta Criticidade / Risco│
      │ LM Studio Local      │  │ API OpenAI Remota (GPT-4o)      │
      │ (Custo marginal: $0) │  │ (Resolução prioritária)         │
      └──────────────────────┘  └─────────────────────────────────┘
```

## 🚀 Funcionalidades

* **Triagem de baixa latência:** Indexação vetorial usando `IndexHNSWFlat` do FAISS sobre embeddings normalizados.

* **Detecção de Fronteiras Ambíguas:** Cálculo de margem de separação (`Δ_margin`) em tempo de execução.

* **Função de Perda Assimétrica (Risk-Aware):** Na dúvida entre dúvida simples e fraude, o sistema prioriza a segurança e o recall da classe crítica.

* **Infraestrutura Local:** Camada de vetorização e SLM executadas localmente via **LM Studio**, compatível com o SDK da OpenAI.

## 🛠️ Tecnologias e Dependências

* Python 3.10+
* LM Studio (servidor local em `http://localhost:1234/v1`)

### Modelos Locais Recomendados

* Embeddings: `nomic-ai/nomic-embed-text-v1.5-GGUF`
* SLM: `meta-llama-3-8b-instruct`

### Bibliotecas Python

* `openai`: Comunicação unificada com o LM Studio e OpenAI
* `faiss-cpu`: Indexação e busca vetorial em grafos aproximados
* `numpy`: Vetorização e manipulação matricial
* `tabulate`: Apresentação estruturada dos benchmarks

## 📦 Instalação

```bash
git clone https://github.com/seu-usuario/smart-llm-router.git
cd smart-llm-router

python3 -m venv .venv
source .venv/bin/activate

pip install openai faiss-cpu numpy tabulate
```

## ⚙️ Configuração do LM Studio

1. Abra o **LM Studio** e inicialize o servidor local em `http://localhost:1234`.

2. Carregue o modelo de embeddings, por exemplo `nomic-embed-text-v1.5`.

3. Carregue o SLM com suporte à extração de logprobs, por exemplo `meta-llama-3-8b-instruct`.

4. Configure sua chave da OpenAI no ambiente:

```bash
export OPENAI_API_KEY="sk-sua-chave-aqui"
```

## 🧪 Como Executar o Benchmark

Para rodar o comparativo experimental entre centróides médios, busca vetorial HNSW pura e a proposta híbrida:

```bash
python main.py
```

### Exemplo de Saída Experimental

| **Cenário de Teste**                          | **Ground Truth** | **Centróide** | **HNSW Puro** | **Híbrido Proposto**  | **Latência Híbrida** |
| --------------------------------------------- | ---------------- | ------------- | ------------- | --------------------- | -------------------: |
| "A agência abre no sábado para atendimento?"  | Simples          | ✅ Acerto      | ✅ Acerto      | ✅ Acerto              |               1.9 ms |
| "Saque não reconhecido na minha conta agora." | Crítico          | ✅ Acerto      | ✅ Acerto      | ✅ Acerto              |               2.1 ms |
| "Não entro no app e recebi SMS com link."     | Crítico          | ❌ Falha       | ❌ Falha       | ✅ Acerto (Gating SLM) |             112.4 ms |
| "Qual a taxa de remessa internacional?"       | Simples          | ❌ Falha       | ✅ Acerto      | ✅ Acerto              |               2.2 ms |

## 📚 Fundamentação Teórica

* **HNSW (Hierarchical Navigable Small World):** MALKOV, Y. A.; YASHUNIN, D. A. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI, 2020.

* **Cascatas de Inferência e Custo de Tokens:** CHEN, L.; ZAHARIA, M.; ZOU, J. *FrugalGPT: How to Use Large Language Models Cheaply and Effectively*. arXiv:2305.05176, 2023.

* **Roteamento Híbrido Preditivo:** DING, D. et al. *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*. ICLR, 2024.

* **Avaliação Sistemática de Roteadores:** HU, Q. et al. *RouterBench: A Benchmark for Multi-LLM Routing System*. arXiv:2403.12031, 2024.

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
