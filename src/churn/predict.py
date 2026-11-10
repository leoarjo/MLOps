"""Consumir o modelo de produção (Encontro 8, slide 21).

Entrypoint::

    python -m churn.predict

Carrega ``models:/churn@champion`` e pontua algumas linhas do conjunto de
teste. É a prova de que o ciclo fechou: o modelo é alcançável por uma
referência estável, sem ninguém saber qual run o gerou.

Este módulo é o ensaio do Encontro 12, quando o consumidor passa a ser uma API
FastAPI em vez de um ``print()``.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from churn.config import Settings, settings
from churn.model import prepare
from churn.registry import champion_uri, load_champion

logger = logging.getLogger(__name__)


def score_sample(rows: int = 5, cfg: Settings = settings) -> pd.DataFrame:
    """Pontua as primeiras linhas do teste com o champion.

    As features passam pelo ``encoder`` antes do ``predict``: o champion é só o
    classificador, e espera a entrada já codificada. Enquanto o encoder não
    estiver dentro do pacote do modelo (um wrapper ``pyfunc``, assunto do
    Encontro 12), é responsabilidade de quem consome aplicar a mesma
    transformação do treino.
    """
    _, x_test, _, y_test, encoder = prepare(cfg)
    features = encoder.transform(x_test.head(rows))

    model = load_champion(cfg)
    predictions = model.predict(features)

    return pd.DataFrame(
        {
            "churn_previsto": ["Yes" if p else "No" for p in predictions],
            "churn_real": ["Yes" if y else "No" for y in y_test.head(rows)],
        },
        index=x_test.head(rows).index,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pontua uma amostra com o champion.")
    parser.add_argument("--rows", type=int, default=5, help="linhas a pontuar")
    args = parser.parse_args()

    result = score_sample(args.rows)
    print(f"modelo: {champion_uri()}")
    print(result.to_string())
    acertos = (result["churn_previsto"] == result["churn_real"]).sum()
    print(f"\n{acertos}/{len(result)} acertos nesta amostra")


if __name__ == "__main__":
    main()
