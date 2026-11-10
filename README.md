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
| 6 | ML Pipeline — tracking | `mlflow` no `train()`, `lineage.py` (tags de dado e commit), grid em `experiments.py` |
| 7 | ML Pipeline — registry | `registry.py`: melhor run → `churn` v1, alias `@champion`, gate de `roc_auc` |
| 8 | ML Pipeline — empacotamento | `signature` + `input_example` + cloudpickle, consumo por `models:/churn@champion` |
| 12 | Serving | `export.py` (champion → bundle), API FastAPI, front Streamlit, `Dockerfile.serve` + deploy no EC2 |

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
│  ├─ model.py       # prepare() + treino rastreado (entrypoint)
│  ├─ lineage.py     # tags de linhagem: md5 do dado e commit do código
│  ├─ experiments.py # o grid de 9 runs (entrypoint)
│  ├─ registry.py    # melhor run -> registro -> alias @champion (entrypoint)
│  ├─ predict.py     # consome models:/churn@champion (entrypoint)
│  ├─ export.py      # champion -> artifacts/champion/ servível (entrypoint)
│  ├─ serving.py     # contrato de entrada (Pydantic) + bundle + inferência
│  ├─ api.py         # FastAPI: /health, /model, /predict, /predict/batch
│  └─ frontend.py    # Streamlit: formulário do cliente + lote por CSV
├─ tests/            # pytest
├─ data/
│  └─ churn.csv.dvc  # ponteiro do dataset (os bytes ficam no DVC)
├─ artifacts/        # modelo treinado (fora do Git, saída do dvc.yaml)
├─ dvc.yaml          # pipeline reproduzível: dado + código -> modelo
├─ dvc.lock          # hashes da última execução do pipeline
├─ mlflow.db         # backend de tracking (fora do Git)
├─ mlruns/           # artefatos dos runs (fora do Git)
├─ Dockerfile        # imagem de treino
├─ compose.yml       # treino com volumes, sem decorar flags
├─ Dockerfile.serve  # imagem de serving: API + front + champion
├─ compose.serve.yml # sobe api (:8000) e front (:8501)
├─ deploy/           # user data do EC2
├─ docs/             # guia de deploy no EC2
└─ pyproject.toml    # dependências (travadas no uv.lock)
```

## Como rodar

Com `uv`:

```bash
uv sync                              # cria o ambiente a partir do pyproject/uv.lock
uv run python -m churn.profiling     # profiling do dataset
uv run python -m churn.model         # valida, treina, registra o run no MLflow
uv run python -m churn.experiments   # o grid de 9 runs
uv run python -m churn.registry      # promove o melhor a @champion
uv run python -m churn.predict       # pontua uma amostra com o champion
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

O run do MLflow gerado dentro do container é **efêmero**: o backend default
(`sqlite:///mlflow.db`) cai em `/app` e morre com o container. Para persistir,
monte uma pasta e aponte o tracking para ela:

```bash
docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/mlruns:/app/mlruns" \
  -e CHURN_TRACKING_URI=sqlite:////app/mlruns/mlflow.db churn:0.1
```

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

## ML Pipeline (Encontros 6, 7 e 8)

### Tracking — cada treino é um run

```bash
uv run python -m churn.model                               # um run
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # localhost:5000
```

O backend é **sqlite**, não `./mlruns`: no MLflow 3 o file store entrou em
manutenção. A URI vive no `Settings` (`CHURN_TRACKING_URI` sobrescreve), em vez
de depender de uma variável de ambiente exportada à mão — é o atrito "os runs
somem" do deck.

Cada run registra:

| | o que vai | de onde vem |
|---|---|---|
| **params** | `n_estimators`, `max_depth`, `test_size`, `random_state` | `Settings.hyperparams()` |
| **metrics** | accuracy, precision, recall, f1, roc_auc | `evaluate()` |
| **tags** | `data_md5`, `git_commit`, `git_dirty` | `lineage.py` |
| **model** | o modelo + signature + exemplo | `mlflow.sklearn.log_model` |

As tags são o que fecha a cadeia dado → código → experimento → modelo: o
`data_md5` sai do ponteiro `data/churn.csv.dvc` (100 bytes de YAML), não do CSV.

### O grid

```bash
uv run python -m churn.experiments   # 3 n_estimators × 3 max_depth = 9 runs
```

`prepare()` roda uma vez e todos os runs compartilham o mesmo split, encoder e
versão do dado — sem isso a comparação não vale. Resultado no dado v2:

| run | n_estimators | max_depth | roc_auc | accuracy |
|---|---|---|---|---|
| rf-n500-d8 | 500 | 8 | 0.8220 | 0.7749 |
| rf-n200-d8 | 200 | 8 | 0.8220 | 0.7727 |
| **rf-n100-d8** | **100** | **8** | **0.8215** | **0.7732** |
| rf-n500-d16 | 500 | 16 | 0.8169 | 0.7682 |
| rf-n200-dNone | 200 | — | 0.8150 | 0.7609 |

`max_depth=8` domina e profundidade 16 já piora (overfit), igual ao deck.

### Registry — promover

```bash
uv run python -m churn.registry      # churn v1 @champion
```

O `promote()` faz, nesta ordem: acha o melhor run concluído → checa o **gate**
(`roc_auc >= 0.80`) → registra → move o alias `@champion` → copia `roc_auc`,
`data_md5` e `git_commit` como tags da versão. Avaliar antes de registrar é o
que evita um modelo ruim marcado como produção, mesmo por um instante.

**Desempate:** as três melhores configurações empatam dentro de 0.001 de
roc_auc. Entre modelos estatisticamente equivalentes, o `best_run()` escolhe o
de menos árvores — `rf-n100-d8`, 5× mais barato de servir que o de 500 por
0.0005 de roc_auc. Promover o topo da lista bruta deixaria a escolha na mão da
sexta casa decimal.

Aliases e não stages: `Staging`/`Production` estão depreciados. Promover é
mover o `@champion`; reverter é mover de volta.

### Consumir

```bash
uv run python -m churn.predict --rows 6
```

```python
model = mlflow.pyfunc.load_model("models:/churn@champion")
model.predict(features)
```

`pyfunc` e não `sklearn`: é o contrato universal (só `predict(df)`). A API do
Encontro 12 carrega o flavor `sklearn` do mesmo pacote porque precisa de
`predict_proba`: ela devolve a probabilidade, não só o 0/1.

⚠️ **O champion espera features já codificadas.** O encoder não faz parte do
objeto sklearn, então vai como artefato `encoder/encoder.pkl` em cada run e
quem consome aplica a mesma transformação do treino. No serving (abaixo) esse
cuidado mora num lugar só: `serving.ModelBundle`.

## Serving (Encontro 12) — API, front e deploy

```bash
uv run python -m churn.export                         # champion -> artifacts/champion/
uv run uvicorn churn.api:app --port 8000              # API  (docs em /docs)
uv run streamlit run src/churn/frontend.py            # front em :8501
# ou os dois em container:
docker compose -f compose.serve.yml up --build
```

```
 Streamlit (:8501) ──registro completo (JSON)──▶ FastAPI (:8000) ──▶ ModelBundle
   formulário / CSV                                 contrato Pydantic       encoder + RF
```

**Export, não registry, em produção.** O `churn.export` resolve o
`models:/churn@champion` e grava uma pasta autocontida: o pacote MLflow como
foi registrado (com signature), o `encoder.pkl` do run e um `metadata.json` com
versão, run, roc_auc, `data_md5` e `git_commit`. A API lê só essa pasta, e a
pasta entra na imagem. O container não depende do `mlflow.db`, e a imagem
identifica sozinha qual modelo está servindo.

**Contrato de entrada.** `serving.CustomerRecord` é o registro inteiro do
cliente, isto é, as 19 colunas do `churn.csv` menos o alvo. As categorias vêm do próprio
`schema.py`, então API e dado de treino não divergem. Um campo fora do
contrato vira **422** apontando campo e valor, antes de o modelo ser tocado.
`customerID` é opcional e volta na resposta. Colunas extras (como `Churn`) são
ignoradas.

**Mesma transformação do treino.** O `ModelBundle.predict` usa as funções de
`features.py` (não uma cópia) e reordena as colunas pelo `feature_names_in_`
do modelo. O teste `test_batch_preserva_ordem_e_bate_com_o_modelo` cobre isso;
no dado real, API e `prepare()` dão a mesma probabilidade nas 1761 linhas de
teste.

| rota | o quê |
|---|---|
| `GET /health` | liveness (healthcheck do compose) |
| `GET /model` | versão do champion, run, linhagem e limiar |
| `POST /predict` | um `CustomerRecord` → `{customerID, churn_probability, churn}` |
| `POST /predict/batch` | `{"records": [...]}` (até 10.000) → lista na mesma ordem |

O **front** não carrega modelo nenhum: monta o registro (com os campos
dependentes coerentes; sem internet → `No internet service`), chama a API e
mostra a probabilidade. Na aba de lote ele aceita um CSV no formato do
`churn.csv` e, se houver coluna `Churn`, compara previsto × real.

**Duas imagens.** `Dockerfile` treina (`--no-default-groups`: sem
FastAPI/Streamlit). `Dockerfile.serve` serve (`--group serve`: sem dvc/pytest)
e é a única que leva `artifacts/champion/`, liberado no
`Dockerfile.serve.dockerignore`. O modelo é a última camada: promover um
champion novo rebuilda em segundos.

Deploy no EC2: [`docs/deploy-ec2.md`](docs/deploy-ec2.md).

## Próximo encontro

- **Bloco 3 (Enc. 9–11)** — Airflow: orquestrar dados → treino → avaliação.
