"""Grid de experimentos rastreados (Encontro 6, slide 12).

Entrypoint::

    python -m churn.experiments

Roda 3 × 3 = 9 variações do Random Forest, cada uma como um run permanente e
comparável do MLflow. É a resposta à dor do slide 5: em vez de 40 ``print()``
e um post-it, as nove configurações ficam no banco, ordenáveis por métrica.

Duas regras tornam a comparação justa:

* ``prepare()`` roda UMA vez — todos os runs veem exatamente o mesmo split,
  o mesmo encoder e a mesma versão do dado;
* ``random_state`` é fixo — a diferença entre dois runs vem do hiperparâmetro,
  não do sorteio das árvores.
"""
from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from churn import lineage
from churn.config import Settings, settings
from churn.evaluate import evaluate
from churn.features import CategoricalEncoder
from churn.model import build_model, prepare

logger = logging.getLogger(__name__)

# O grid do slide 12. max_depth=None deixa as árvores crescerem até o fim.
N_ESTIMATORS = (100, 200, 500)
MAX_DEPTHS: tuple[int | None, ...] = (None, 8, 16)


def log_model_with_contract(
    model, features: pd.DataFrame, encoder: CategoricalEncoder
) -> None:
    """Registra o modelo já com contrato — signature, exemplo e o encoder.

    O deck (slide 35) acrescenta a ``signature`` só no modelo vencedor. Aqui
    todos os runs saem com contrato, por um motivo prático: não se sabe de
    antemão qual deles vai ganhar, e um modelo sem schema não é promovível.

    O ``encoder`` vai como artefato ao lado do modelo porque ele NÃO faz parte
    do objeto sklearn: quem carregar o champion recebe um modelo que espera
    features já codificadas. Sem o encoder, "Month-to-month" não vira o mesmo
    número que o treino viu.
    """
    signature = infer_signature(features, model.predict(features))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=features.head(3),
        serialization_format="cloudpickle",
    )

    # O encoder não é um estimador sklearn (é a nossa classe do Encontro 2),
    # então vai como artefato pickle em vez de flavor.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "encoder.pkl"
        path.write_bytes(pickle.dumps(encoder))
        mlflow.log_artifact(str(path), artifact_path="encoder")


def run_grid(cfg: Settings = settings) -> pd.DataFrame:
    """Executa o grid e devolve os runs ordenados pela métrica principal."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    x_train, x_test, y_train, y_test, encoder = prepare(cfg)
    x_train_enc = encoder.transform(x_train)
    x_test_enc = encoder.transform(x_test)
    tags = lineage.tags()

    for n_estimators in N_ESTIMATORS:
        for max_depth in MAX_DEPTHS:
            variant = cfg.model_copy(
                update={"n_estimators": n_estimators, "max_depth": max_depth}
            )
            run_name = f"rf-n{n_estimators}-d{max_depth}"

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(tags)
                mlflow.log_params(variant.hyperparams())

                model = build_model(variant).fit(x_train_enc, y_train)
                metrics = evaluate(model, x_test_enc, y_test)
                mlflow.log_metrics(metrics)
                log_model_with_contract(model, x_train_enc, encoder)

            logger.info("%s -> roc_auc %.4f", run_name, metrics["roc_auc"])

    return mlflow.search_runs(
        experiment_names=[cfg.experiment_name],
        order_by=["metrics.roc_auc DESC"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runs = run_grid()
    columns = ["tags.mlflow.runName", "params.n_estimators", "params.max_depth",
               "metrics.roc_auc", "metrics.accuracy"]
    print("\nRuns ordenados por roc_auc:")
    print(runs[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
