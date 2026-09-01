"""Avaliação do modelo.

Devolve as métricas como dicionário, sem imprimir nada. Quem decide o que
fazer com elas é quem chamou — no Encontro 6 é o MLflow que vai consumir
exatamente este dicionário.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    model: ClassifierMixin, features: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    """Calcula as métricas de classificação binária.

    Só ``accuracy`` (a única do script original) engana num dataset com ~26%
    de churn: um modelo que chuta "No" para todo mundo acerta 74%. Por isso
    ``recall`` e ``roc_auc`` entram aqui.
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
