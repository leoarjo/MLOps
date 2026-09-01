"""Engenharia de features.

Transformações sem estado são funções puras. O encoding de categóricas guarda
estado — as categorias vistas — e por isso vive numa classe com ``fit`` e
``transform``: aprende SOMENTE no treino e é aplicado no teste.

É essa separação que corrige o vazamento de dados do script original, onde o
``LabelEncoder`` era ajustado sobre o dataset inteiro (treino + teste juntos).

Nota de projeto: não há normalização de escala aqui, de propósito. Random
Forest particiona por limiar e é indiferente à escala das features — dividir
por 118.0, 8600.0 e 72.0 no script original não fazia nada além de esconder
os valores reais.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as features derivadas.

    ``gasto_por_mes`` é row-wise (depende só da própria linha), então pode ser
    calculada antes do split sem risco de vazamento. O ``+ 1`` no denominador
    protege contra ``tenure == 0``.
    """
    out = df.copy()
    out["gasto_por_mes"] = out["TotalCharges"] / (out["tenure"] + 1)
    return out


def split_features_target(
    df: pd.DataFrame, target: str, positive_label: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) de y (alvo binário 0/1).

    O mapeamento é explícito: ``positive_label`` vira 1, o resto vira 0. No
    script original o alvo era codificado pelo mesmo ``LabelEncoder`` das
    features e só funcionou por acaso — em ordem alfabética "No" caiu em 0 e
    "Yes" em 1.
    """
    features = df.drop(columns=[target])
    labels = (df[target] == positive_label).astype(int)
    return features, labels


class CategoricalEncoder:
    """Codifica colunas categóricas em inteiros, sem vazamento.

    ``fit`` aprende as categorias apenas com o conjunto de treino;
    ``transform`` aplica o mesmo mapeamento a treino e teste. Categorias
    inéditas no teste viram ``-1`` em vez de quebrar.
    """

    def __init__(self) -> None:
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame) -> "CategoricalEncoder":
        # "object" cobre pandas 2.x; "string" cobre o novo dtype do pandas 3.x
        self._columns = features.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
        self._encoder.fit(features[self._columns])
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        out[self._columns] = self._encoder.transform(out[self._columns])
        return out

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
