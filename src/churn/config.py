"""Configuração central do projeto.

Todo caminho, hiperparâmetro e constante que estava espalhado pelo
``train_churn.py`` vive aqui — nunca no meio da lógica.

Qualquer valor pode ser sobrescrito por variável de ambiente com o prefixo
``CHURN_`` (ex.: ``CHURN_DATA_PATH=/app/data/churn.csv``) ou por um arquivo
``.env``. É isso que vai permitir, no Encontro 3, rodar o mesmo código dentro
do container sem alterar uma linha.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros do pipeline de churn."""

    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        protected_namespaces=(),
    )

    # --- dados ---
    data_path: Path = Path("data/churn.csv")
    id_column: str = "customerID"
    target: str = "Churn"
    positive_label: str = "Yes"

    # --- split e modelo ---
    test_size: float = 0.25
    random_state: int = 42
    n_estimators: int = 200

    # --- saída ---
    artifact_path: Path = Path("artifacts/churn_model.pkl")


settings = Settings()
