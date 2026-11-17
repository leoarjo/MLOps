"""Model Registry: achar o melhor run, promover e consumir (Encontros 7 e 8).

Entrypoint::

    python -m churn.registry

Um run é um experimento; um *registered model* é um artefato de produção. A
diferença é a dor do slide 17: semanas depois ninguém lembra qual dos 40 runs
era o bom. O registry resolve isso com um nome ("churn"), versões imutáveis e
um **alias** móvel — ``@champion`` — que diz qual versão serve.

Promover = mover o alias. Reverter = mover de volta. Nada de deploy manual.
"""
from __future__ import annotations

import logging

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from churn.config import Settings, settings

logger = logging.getLogger(__name__)

# Diferenças de roc_auc menores que isto são ruído, não mérito: com elas o
# "melhor" run passaria a depender da sexta casa decimal.
TIE_TOLERANCE = 1e-3

MAIN_METRIC = "metrics.roc_auc"


def passes_gate(metrics: dict[str, float], min_roc_auc: float) -> bool:
    """O portão do slide 28: sem o limiar mínimo, não promove.

    Função pura de propósito — é a regra de negócio da promoção, e é o que os
    testes exercitam sem precisar de um servidor MLflow de pé.
    """
    return metrics.get("roc_auc", float("-inf")) >= min_roc_auc


def search_finished_runs(cfg: Settings = settings) -> pd.DataFrame:
    """Runs concluídos do experimento, do melhor roc_auc para o pior.

    O filtro por ``FINISHED`` não é detalhe: um run que quebrou no meio fica
    no banco com métricas e **sem** o modelo salvo. Promover esse run falharia
    só na hora de registrar — tarde demais.
    """
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return mlflow.search_runs(
        experiment_names=[cfg.experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"{MAIN_METRIC} DESC"],
    )


def best_run(cfg: Settings = settings) -> pd.Series:
    """O melhor run — com desempate pelo modelo mais barato.

    O ``order_by`` do MLflow já devolve o maior roc_auc primeiro, mas no nosso
    grid as duas melhores configurações empatam na quarta casa (0.8220). Entre
    modelos estatisticamente equivalentes, escolher o de menos árvores é o que
    um time faria: menor, mais rápido de servir e mais barato — sem perda
    mensurável de qualidade.
    """
    runs = search_finished_runs(cfg)
    if runs.empty:
        raise ValueError(
            f"nenhum run concluido no experimento '{cfg.experiment_name}'. "
            "Rode `python -m churn.experiments` primeiro."
        )

    top = runs[MAIN_METRIC].max()
    tied = runs[runs[MAIN_METRIC] >= top - TIE_TOLERANCE].copy()
    tied["_n"] = pd.to_numeric(tied["params.n_estimators"], errors="coerce")
    chosen = tied.sort_values(["_n", MAIN_METRIC], ascending=[True, False]).iloc[0]

    if len(tied) > 1:
        logger.info(
            "%d runs empatados dentro de %.0e; escolhido o de menos arvores (%s)",
            len(tied),
            TIE_TOLERANCE,
            chosen["params.n_estimators"],
        )
    return chosen


def promote(cfg: Settings = settings) -> ModelVersion:
    """Registra o melhor run e aponta ``@champion`` para ele.

    A ordem importa: avaliar, **depois** registrar. Registrar primeiro e medir
    depois deixaria um modelo ruim marcado como produção no intervalo.
    """
    run = best_run(cfg)
    metrics = {"roc_auc": float(run[MAIN_METRIC])}

    if not passes_gate(metrics, cfg.min_roc_auc):
        raise ValueError(
            f"roc_auc {metrics['roc_auc']:.4f} abaixo do limiar "
            f"{cfg.min_roc_auc:.2f} — nao promove"
        )

    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    run_id = run["run_id"]

    # No MLflow 3 o modelo não é mais um arquivo dentro de
    # ``artifacts/model``: é um *logged model* de primeira classe, com id
    # próprio, ligado ao run por ``outputs.model_outputs``. A URI
    # ``runs:/<id>/model`` do slide 19 continua resolvendo — mas quem procura
    # o modelo com ``list_artifacts`` não acha nada e conclui errado.
    outputs = client.get_run(run_id).outputs
    if not (outputs and outputs.model_outputs):
        raise ValueError(
            f"o run {run_id} terminou sem modelo salvo — nada a promover"
        )

    version = mlflow.register_model(
        f"runs:/{run_id}/model", cfg.registered_model_name
    )
    client.set_registered_model_alias(
        cfg.registered_model_name, cfg.champion_alias, version=version.version
    )
    # A linhagem do slide 21 fica legível no próprio registry, sem abrir o run.
    client.set_model_version_tag(
        cfg.registered_model_name, version.version, "roc_auc", f"{metrics['roc_auc']:.4f}"
    )
    for tag in ("data_md5", "git_commit"):
        value = run.get(f"tags.{tag}")
        if isinstance(value, str):
            client.set_model_version_tag(
                cfg.registered_model_name, version.version, tag, value
            )

    logger.info(
        "%s v%s @%s (roc_auc %.4f, run %s)",
        cfg.registered_model_name,
        version.version,
        cfg.champion_alias,
        metrics["roc_auc"],
        run_id,
    )
    return version


def champion_uri(cfg: Settings = settings) -> str:
    """``models:/churn@champion`` — a referência estável de produção."""
    return f"models:/{cfg.registered_model_name}@{cfg.champion_alias}"


def load_champion(cfg: Settings = settings):
    """Carrega o modelo de produção pelo flavor ``pyfunc``.

    ``pyfunc`` e não ``sklearn`` de propósito: é o contrato universal, com só
    um ``predict(df)``. É o que a API do Encontro 12 vai consumir, sem saber
    que por baixo existe um Random Forest.
    """
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return mlflow.pyfunc.load_model(champion_uri(cfg))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    version = promote()
    print(
        f"{settings.registered_model_name} v{version.version} "
        f"@{settings.champion_alias} — pronto para consumo em {champion_uri()}"
    )


if __name__ == "__main__":
    main()
