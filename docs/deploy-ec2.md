# Deploy no AWS EC2 — API + Streamlit

```
 seu computador                                   EC2 (Ubuntu 24.04)
 ─────────────────────────────                    ──────────────────────────────────
 experiments → registry → export                  docker compose -f compose.serve.yml
        │                                           ┌──────────────┐   ┌──────────────┐
        ▼                                           │ front :8501  │──▶│  api :8000   │
 artifacts/champion/  ── scp ──────────────────▶    │  Streamlit   │   │   FastAPI    │
 (modelo + encoder + metadata)                      └──────────────┘   │ + champion   │
                                                                       └──────────────┘
```

O modelo **não** é treinado no EC2. Quem treina e promove é você, na sua
máquina, onde estão o `mlflow.db` e o dado do DVC. O EC2 recebe só o bundle
exportado (`artifacts/champion/`, ~5 MB), e ele entra na imagem Docker.

## 1. Na sua máquina: gerar o bundle do champion

```bash
uv sync
uv run dvc pull                      # garante o churn.csv da versão certa
uv run python -m churn.experiments   # grid de 9 runs (se ainda não rodou)
uv run python -m churn.registry      # melhor run -> churn vN @champion
uv run python -m churn.export        # -> artifacts/champion/
```

Teste local antes de subir (precisa do Docker Desktop):

```bash
docker compose -f compose.serve.yml up --build
# front: http://localhost:8501    API: http://localhost:8000/docs
```

## 2. Criar a instância

No console da AWS: **EC2 → Launch instance**.

| campo | valor |
|---|---|
| AMI | **Ubuntu Server 24.04 LTS** (x86_64) |
| tipo | `t3.small` (2 GB). `t3.micro` (free tier, 1 GB) funciona graças ao swap do user data, só que o build fica lento |
| key pair | crie uma (ex.: `churn.pem`) e guarde o arquivo |
| disco | **20 GB** gp3. O padrão de 8 GB fica apertado com a imagem (~1,5 GB) mais o cache de build |
| security group | regras de entrada abaixo |
| user data | *Advanced details → User data*: cole o conteúdo de [`deploy/ec2-user-data.sh`](../deploy/ec2-user-data.sh) |

Security group (entrada):

| porta | origem | para quê |
|---|---|---|
| 22 | **My IP** | SSH |
| 8501 | `0.0.0.0/0` (ou My IP) | front Streamlit |
| 8000 | My IP (opcional) | API e `/docs` direto |

O user data instala o Docker e clona o repositório em `~/MLOps`. Espere
uns 2–3 minutos depois que a instância ficar *running*. Para conferir se ele
terminou:

```bash
ssh -i churn.pem ubuntu@<IP-PUBLICO> "tail -5 /var/log/cloud-init-output.log; docker --version"
```

> **Windows (PowerShell):** se o `ssh` reclamar de `UNPROTECTED PRIVATE KEY FILE`,
> restrinja a permissão da chave:
> ```powershell
> icacls churn.pem /inheritance:r
> icacls churn.pem /grant:r "$($env:USERNAME):(R)"
> ```

## 3. Enviar o bundle e subir

Da raiz do repositório, na sua máquina (os comandos são os mesmos no PowerShell):

```bash
scp -i churn.pem -r artifacts/champion ubuntu@<IP-PUBLICO>:~/MLOps/artifacts/
```

No EC2:

```bash
ssh -i churn.pem ubuntu@<IP-PUBLICO>
cd ~/MLOps
git pull                                         # código mais recente da main
docker compose -f compose.serve.yml up -d --build
docker compose -f compose.serve.yml ps           # api deve ficar "healthy"
```

Pronto:

- front: `http://<IP-PUBLICO>:8501`
- API: `http://<IP-PUBLICO>:8000/docs` (se a porta 8000 estiver liberada)

Os containers têm `restart: unless-stopped`, então voltam sozinhos se a
instância reiniciar.

## 4. Promover um modelo novo

```bash
# na sua máquina
uv run python -m churn.registry && uv run python -m churn.export
scp -i churn.pem -r artifacts/champion ubuntu@<IP-PUBLICO>:~/MLOps/artifacts/

# no EC2
docker compose -f compose.serve.yml up -d --build
```

Só a última camada da imagem (o `COPY artifacts/champion/`) é refeita, então o
rebuild leva segundos. A barra lateral do Streamlit mostra a versão nova.

## Alternativa: imagem pronta no Docker Hub

Em vez de buildar no EC2, builde na sua máquina e publique a imagem. Assim o
artefato que vai para produção é exatamente o que você testou.

```bash
# na sua máquina (--platform importa se você estiver num Mac M1/M2/M3)
docker build --platform linux/amd64 -f Dockerfile.serve -t <usuario>/churn-serve:v1 .
docker push <usuario>/churn-serve:v1

# no EC2 (não precisa do bundle via scp: ele já está dentro da imagem)
export CHURN_IMAGE=<usuario>/churn-serve:v1
docker compose -f compose.serve.yml pull
docker compose -f compose.serve.yml up -d --no-build
```

Com o ECR o fluxo é o mesmo; muda o login (`aws ecr get-login-password ...`)
e a instância precisa de uma IAM role com `AmazonEC2ContainerRegistryReadOnly`.

## Problemas comuns

| sintoma | causa provável |
|---|---|
| build falha em `COPY artifacts/champion/` | o bundle não foi enviado/gerado: passo 1 e o `scp` do passo 3 |
| `api` reiniciando em loop | `docker compose -f compose.serve.yml logs api`. Bundle incompleto aparece com a mensagem "Rode `uv run python -m churn.export`" |
| front abre mas diz "API indisponível" | a `api` ainda está subindo (~20 s, o MLflow demora a importar) ou caiu; veja `ps` e `logs api` |
| página não abre no navegador | porta 8501 fora do security group, ou você está usando `https://` (é `http://`) |
| build morre sem mensagem numa t3.micro | falta de memória: confira o swap com `free -h` |
| IP mudou | o IP público muda a cada *stop/start*. Para um fixo, associe um Elastic IP (que é cobrado com a instância parada) |

**Custo:** pare a instância (*Instance state → Stop*) quando não estiver usando.
Disco parado continua sendo cobrado, só que bem menos. *Terminate* apaga tudo.
