"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrypoint do projeto::

    python -m churn.model

No Encontro 3, dentro do container, vira ``uv run python -m churn.model``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa.

    ``random_state`` é o que faz a acurácia parar de mudar entre execuções —
    o "acc = 0.79 ??? (ontem tinha dado 0.80)" do script original.
    """
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
    )


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
    """Pipeline completo: dados -> features -> split -> encode -> treino -> avaliação."""
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
    model = build_model(cfg).fit(encoder.transform(x_train), y_train)
    metrics = evaluate(model, encoder.transform(x_test), y_test)

    save_artifact(model, encoder, cfg.artifact_path)
    return model, metrics


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no conjunto de teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
