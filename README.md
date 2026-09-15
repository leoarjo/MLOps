# MLOps — projeto churn

Repositório da disciplina de MLOps (IESB). Projeto-fio-condutor: previsão de
**churn** (classificação binária tabular).

Estado do projeto por encontro:

| Encontro | Tema | O que entrou no projeto |
|---|---|---|
| 2 | Git e estrutura | `train_churn.py` → módulos em `src/churn/`, config Pydantic, `pyproject.toml` + `uv.lock`, testes |
| 3 | Docker | `Dockerfile` (uv), `.dockerignore`, `compose.yml` com volumes |
| 4 | Data Pipeline — validação | profiling, contratos pandera (`schema.py`) no `data.py`, testes que quebram o dado de propósito |
| 5 | Data Pipeline — DVC | `churn.csv` fora do Git (ponteiro `.dvc`), remote local, pipeline `dvc.yaml` |

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
├─ data/
│  └─ churn.csv.dvc  # ponteiro do dataset (os bytes ficam no DVC)
├─ artifacts/        # modelo treinado (fora do Git, saída do dvc.yaml)
├─ dvc.yaml          # pipeline reproduzível: dado + código -> modelo
├─ dvc.lock          # hashes da última execução do pipeline
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

## Versionamento de dados (Encontro 5 — DVC)

O `data/churn.csv` **não está mais no Git**. O Git guarda só o ponteiro
`data/churn.csv.dvc` (md5 + tamanho); os bytes moram no cache `.dvc/cache` e
no remote. O DVC entra como dependência de desenvolvimento (`uv sync` instala).

Remote configurado: `local`, uma pasta **ao lado** do repositório
(`../dvc-remote`). Depois de clonar, crie o remote ou aponte para outro:

```bash
uv run dvc pull                                   # baixa o churn.csv exato do ponteiro
uv run dvc remote modify local url /outro/caminho # se o remote estiver em outro lugar
```

### Pipeline reproduzível

`dvc.yaml` descreve o treino: depende de `data/churn.csv` e `src/churn`, produz
`artifacts/churn_model.pkl`. O `dvc.lock` registra os hashes da última execução.

```bash
uv run dvc repro     # só re-treina se dado ou código mudaram
uv run dvc status    # mostra o que está desatualizado
uv run dvc push      # envia dado e modelo para o remote
```

### Versões do dado e alternância (tarefa de casa)

A v2 acrescentou 100 clientes ao fim do CSV (IDs `14043` a `14142`). O commit
`dados v2` amarra o ponteiro novo **e** o `dvc.lock` do modelo re-treinado —
é a linhagem dado → código → modelo do slide 11.

| versão | clientes | md5 | commit | modelo (accuracy · recall · roc_auc) |
|---|---|---|---|---|
| v1 | 7.043 | `d390bd07b5514a2256a2396993b8b0e3` | `05ca963` | 0.7570 · 0.5602 · 0.8053 |
| v2 | 7.143 | `ca5c74c93b9748bed30992d64be8ce7c` | `8cee22b` (`dados v2`) | 0.7609 · 0.5704 · 0.8150 |

```bash
# registrar uma nova versão depois de alterar o data/churn.csv
uv run dvc add data/churn.csv
uv run dvc repro                   # re-treina e atualiza o dvc.lock
git commit -am "dados vN"
uv run dvc push

# voltar para a v1 (acha o commit com: git log --oneline -- data/churn.csv.dvc)
git checkout 05ca963 -- data/churn.csv.dvc
uv run dvc checkout                # churn.csv volta a ter 7.043 clientes

# retornar para a versão do commit atual (v2)
git checkout HEAD -- data/churn.csv.dvc
uv run dvc checkout
```

Regra: **`dvc push` sempre depois do commit** — senão o Git tem o ponteiro e o
remote não tem os bytes.

## Próximo encontro

- **Enc. 6** — MLflow: rastrear experimentos (o ML Pipeline acende).
