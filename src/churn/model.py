"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrypoint do projeto::

    python -m churn.model

No Encontro 3, dentro do container, vira ``uv run python -m churn.model``.

A partir do Encontro 6 cada execução também é um *run* do MLflow: params,
métricas, tags de linhagem e o modelo empacotado ficam registrados. O
``print()`` continua, mas já não é o único registro do resultado — era ele
que fazia o "rodei 40 experimentos e perdi o melhor" do slide 5.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn import lineage
from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)

logger = logging.getLogger(__name__)

Split = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, CategoricalEncoder]


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa.

    ``random_state`` é o que faz a acurácia parar de mudar entre execuções —
    o "acc = 0.79 ??? (ontem tinha dado 0.80)" do script original. Ele também
    é o que torna o grid do Encontro 6 uma comparação justa: a diferença entre
    dois runs vem do hiperparâmetro, não do sorteio.
    """
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )


def prepare(cfg: Settings = settings) -> Split:
    """Dado validado -> features -> split -> encoder ajustado no treino.

    Extraído do ``train()`` no Encontro 6 porque o grid de experimentos precisa
    exatamente dos mesmos dados: preparar uma vez e reutilizar em todos os runs
    é o que garante que as 9 configurações sejam comparáveis entre si (e evita
    revalidar o CSV nove vezes).
    """
    frame = add_derived_features(
        load_clean(cfg.data_path, cfg.id_column, cfg.target)
    )
    features, labels = split_features_target(frame, cfg.target, cfg.positive_label)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    # fit SÓ no treino — é aqui que o vazamento é evitado.
    encoder = CategoricalEncoder().fit(x_train)
    return x_train, x_test, y_train, y_test, encoder


def save_artifact(
    model: RandomForestClassifier, encoder: CategoricalEncoder, path: Path
) -> None:
    """Persiste modelo + encoder juntos.

    O encoder ajustado faz parte do artefato: sem ele a inferência não sabe
    traduzir "Month-to-month" para o mesmo número visto no treino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder}, handle)


def train(cfg: Settings = settings) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Treina, avalia e registra o run no MLflow.

    O ``with start_run`` fecha o run sozinho — inclusive se o treino levantar
    exceção, e aí o run fica marcado como FAILED em vez de aberto para sempre.
    """
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    x_train, x_test, y_train, y_test, encoder = prepare(cfg)

    with mlflow.start_run(run_name="rf") as run:
        mlflow.set_tags(lineage.tags())
        mlflow.log_params(cfg.hyperparams())

        model = build_model(cfg).fit(encoder.transform(x_train), y_train)
        metrics = evaluate(model, encoder.transform(x_test), y_test)

        mlflow.log_metrics(metrics)
        # ``serialization_format`` é o atrito do slide 36: o default do MLflow 3
        # é skops, que recusa salvar `sklearn.tree._tree.Tree` (o nó das
        # árvores) por ser um tipo "não confiável". cloudpickle salva o objeto
        # como ele é — o custo é ficar amarrado às versões das libs, que no
        # nosso caso o uv.lock e a imagem Docker já travam.
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )
        logger.info("run %s registrado em %s", run.info.run_id, cfg.tracking_uri)

    save_artifact(model, encoder, cfg.artifact_path)
    return model, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _, metrics = train()
    print("Modelo treinado. Métricas no conjunto de teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
