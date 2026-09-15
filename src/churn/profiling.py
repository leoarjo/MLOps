"""Profiling do dataset — conhecer o dado antes de escrever o contrato.

Entrypoint::

    python -m churn.profiling

Imprime o retrato que embasou as regras do ``schema.py``: tipos, nulos e
brancos, faixas das numéricas, cardinalidade das categóricas, balanço do
alvo e a relação ``TotalCharges ≈ tenure × MonthlyCharges``. Só pandas — para
um relatório HTML completo, ``ydata-profiling`` faz o mesmo em mais detalhe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import settings
from churn.data import load_raw


def profile(df: pd.DataFrame, target: str) -> str:
    """Monta o relatório de profiling como texto."""
    blanks = df.apply(lambda col: col.astype(str).str.strip().eq("").sum())
    overview = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "brancos": blanks,
            "distintos": df.nunique(),
        }
    )

    numeric = df.select_dtypes("number")
    categorical = df.select_dtypes(exclude="number")

    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    ratio = (total / (df["tenure"] * df["MonthlyCharges"]))[df["tenure"] > 0]

    sections = [
        f"linhas: {len(df)}   colunas: {df.shape[1]}",
        "== visão geral ==\n" + overview.to_string(),
        "== numéricas ==\n" + numeric.describe().T.to_string(),
        "== categóricas (valores) ==\n"
        + "\n".join(
            f"{col}: {sorted(categorical[col].dropna().unique().tolist())}"
            for col in categorical.columns
            if categorical[col].nunique() <= 10
        ),
        "== alvo ==\n" + df[target].value_counts(normalize=True).round(4).to_string(),
        "== TotalCharges / (tenure × MonthlyCharges), tenure > 0 ==\n"
        + ratio.describe().to_string(),
        "brancos em TotalCharges por tenure: "
        + str(df.loc[total.isna(), "tenure"].value_counts().to_dict()),
    ]
    return "\n\n".join(sections)


def main(path: Path = settings.data_path) -> None:
    print(profile(load_raw(path), settings.target))


if __name__ == "__main__":
    main()
