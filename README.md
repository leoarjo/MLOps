# MLOps — projeto churn

Repositório da disciplina de MLOps (IESB). Projeto-fio-condutor: previsão de
**churn** (classificação binária tabular).

Esta branch (`refactor/estrutura`) é a migração do Encontro 2 — o script
monolítico `train_churn.py` reorganizado em módulos.

## Estrutura

```
MLOps/
├─ src/churn/
│  ├─ config.py      # Settings (Pydantic) — caminhos e hiperparâmetros
│  ├─ data.py        # carga + validação + limpeza
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ evaluate.py    # métricas
│  └─ model.py       # treino (entrypoint)
├─ tests/            # pytest
├─ data/             # dataset (versionado por DVC a partir do Enc. 5)
├─ artifacts/        # modelo treinado (fora do Git)
└─ pyproject.toml    # dependências travadas
```

## Como rodar

Com `uv`:

```bash
uv sync                          # cria o ambiente a partir do pyproject/uv.lock
uv run python -m churn.model     # treina e imprime as métricas
uv run pytest                    # roda os testes
```

Sem `uv`:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m churn.model
pytest
```

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded (`C:/Users/ana/Desktop/`) | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure=0`) |
| `dropna()` genérico | validação explícita que falha alto e cedo |
| normalização por números mágicos | removida — Random Forest não é sensível à escala |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa no split e no modelo |
| só `accuracy` | accuracy, precision, recall, f1, roc_auc |
| `modelo_final_v3_ok.pkl` na raiz | `artifacts/churn_model.pkl` (modelo + encoder) |
| tudo em nível de módulo | funções puras + orquestração em `train()` |

## Próximos encontros

- **Enc. 3** — containerizar (Dockerfile com uv).
- **Enc. 4** — schema pandera em `src/churn/schema.py`.
- **Enc. 5** — versionar `data/` com DVC.
