# Imagem oficial do uv, que já traz Python 3.11 + uv instalados.
# Alternativa: FROM python:3.11-slim + COPY --from=ghcr.io/astral-sh/uv /uv /bin/
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Evita o warning de hardlink do uv quando o cache e o venv ficam em
# sistemas de arquivos diferentes (o caso normal dentro de container).
ENV UV_LINK_MODE=copy

WORKDIR /app

# --- camada 1: dependências ---
# Copiar SÓ o manifesto e o lock primeiro. Enquanto eles não mudarem, o Docker
# reaproveita esta camada do cache e não reinstala nada — que é a razão de o
# código vir depois. --no-install-project instala as deps sem o próprio churn.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# --- camada 2: o projeto ---
# README.md é obrigatório aqui: o pyproject.toml declara readme = "README.md",
# e o hatchling recusa a build do wheel se o arquivo não estiver no contexto.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# data/ e artifacts/ NÃO entram na imagem — chegam por volume no docker run.
# Isso mantém a imagem pequena e o dataset fora do artefato distribuído.

CMD ["uv", "run", "python", "-m", "churn.model"]
