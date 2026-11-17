"""Linhagem do experimento — de onde veio o dado e o código (Encontro 6).

O MLflow registra params e métricas sozinho, mas não sabe *qual* dado e *qual*
commit produziram aquele run. São estas duas tags que fecham a cadeia do slide
21: dado (md5 do DVC) → código (commit do Git) → experimento (run) → modelo.

Sem elas o run é um número solto: dois runs com os mesmos hiperparâmetros e
métricas diferentes ficam inexplicáveis.

As duas funções devolvem ``"unknown"`` em vez de quebrar quando a informação
não existe — dentro do container não há ``.git``, e o treino não pode falhar
por causa de uma tag de metadado.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"


def data_md5(pointer: Path = Path("data/churn.csv.dvc")) -> str:
    """Lê o md5 do dado no ponteiro do DVC — sem abrir o CSV.

    O ponteiro é um YAML de ~100 bytes versionado no Git (Encontro 5), então a
    tag sai de graça: não precisa recalcular o hash de um arquivo de 1 MB.
    """
    try:
        outs = yaml.safe_load(pointer.read_text())["outs"]
        return str(outs[0]["md5"])
    except (OSError, KeyError, IndexError, TypeError, yaml.YAMLError):
        logger.warning("nao foi possivel ler o md5 do dado em %s", pointer)
        return UNKNOWN


def git_commit() -> str:
    """O commit curto do HEAD, ou ``"unknown"`` fora de um repositório."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        logger.warning("nao foi possivel obter o commit do git")
        return UNKNOWN
    return done.stdout.strip() or UNKNOWN


def git_dirty() -> bool:
    """``True`` se há mudança não commitada no working tree.

    Um run marcado ``git_dirty=true`` não é reproduzível a partir do commit:
    o código que rodou não é exatamente o que está versionado.
    """
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(done.stdout.strip())


def tags(pointer: Path = Path("data/churn.csv.dvc")) -> dict[str, str]:
    """As tags de linhagem, prontas para ``mlflow.set_tags``."""
    return {
        "data_md5": data_md5(pointer),
        "git_commit": git_commit(),
        "git_dirty": str(git_dirty()).lower(),
    }
