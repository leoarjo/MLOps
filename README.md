# MLOps — projeto churn

Repositório da disciplina de MLOps (IESB). Projeto-fio-condutor: previsão de
**churn** (classificação binária tabular).

Esta branch (`refactor/estrutura`) reúne o Bloco 0 e a primeira metade do
Bloco 1:

| Encontro | Tema | O que entrou no projeto |
|---|---|---|
| 2 | Git e estrutura | `train_churn.py` → módulos em `src/churn/`, config Pydantic, `pyproject.toml` + `uv.lock`, testes |
| 3 | Docker | `Dockerfile` (uv), `.dockerignore`, `compose.yml` com volumes |
| 4 | Data Pipeline — validação | profiling, contratos pandera (`schema.py`) no `data.py`, testes que quebram o dado de propósito |

## Estrutura

```
MLOps/
├─ src/churn/
│  ├─ config.py      # Settings (Pydantic) — caminhos e hiperparâmetros
│  ├─ data.py        # carga + validação + limpeza
│  ├─ schema.py      # contratos de dados (pandera): cru e limpo
│  ├─ profiling.py   # retrato do dataset que embasou o schema
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ evaluate.py    # métricas
│  └─ model.py       # treino (entrypoint)
├─ tests/            # pytest
├─ data/             # dataset
├─ artifacts/        # modelo treinado (fora do Git)
├─ Dockerfile        # imagem de treino
├─ compose.yml       # treino com volumes, sem decorar flags
└─ pyproject.toml    # dependências (travadas no uv.lock)
```

## Como rodar

Com `uv`:

```bash
uv sync                              # cria o ambiente a partir do pyproject/uv.lock
uv run python -m churn.profiling     # profiling do dataset
uv run python -m churn.model         # valida, treina e imprime as métricas
uv run pytest                        # roda os testes
```

Sem `uv`:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e . pytest
python -m churn.model
pytest
```

Com Docker (Encontro 3):

```bash
docker compose up --build            # ou:
docker build -t churn:0.1 .
docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/artifacts:/app/artifacts" churn:0.1
```

`data/` e `artifacts/` não entram na imagem: chegam por volume, e o `.pkl`
treinado aparece em `artifacts/` no host.

## Contrato de dados (Encontro 4)

A validação acontece em dois pontos do `load_clean`:

1. **`RawChurnSchema`** — o CSV como chega: `customerID` único, `tenure` 0–72,
   `Churn` ∈ {Yes, No}. Sem coerção, para não mascarar sujeira.
2. **`ChurnSchema`** — o dado limpo, antes de virar feature: tipos, faixas,
   categorias permitidas de todas as colunas e a relação
   `TotalCharges ≈ tenure × MonthlyCharges`.

Tipos, faixas e categorias são **erro**: o pipeline para antes do treino e os
`failure_cases` (coluna, regra, valor, linha) vão para o log. A relação entre
colunas é **alerta** (`SchemaWarning`): é regra de negócio aproximada — pede
investigação, não derruba o treino.

Para ver o contrato pegando dado quebrado, basta adulterar uma linha do CSV
(ex.: `tenure = 999`) e rodar o treino; os testes em `tests/test_schema.py`
fazem o mesmo de forma automatizada.

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded (`C:/Users/ana/Desktop/`) | `config.py` (Pydantic, sobrescrevível por env `CHURN_*`) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure=0`) |
| `dropna()` genérico | contratos pandera que falham alto e cedo |
| normalização por números mágicos | removida — Random Forest não é sensível à escala |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa no split e no modelo |
| só `accuracy` | accuracy, precision, recall, f1, roc_auc |
| `modelo_final_v3_ok.pkl` na raiz | `artifacts/churn_model.pkl` (modelo + encoder) |
| tudo em nível de módulo | funções puras + orquestração em `train()` |

## Próximo encontro

- **Enc. 5** — versionar `data/churn.csv` com DVC (branch `feat/dvc`).
