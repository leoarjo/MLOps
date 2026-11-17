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
    # ``None`` = árvores crescem até o fim (o comportamento até o Encontro 5).
    # Quem decide o melhor valor é o grid do Encontro 6, não um palpite aqui.
    max_depth: int | None = None

    # --- saída ---
    artifact_path: Path = Path("artifacts/churn_model.pkl")

    # --- MLflow (Encontro 6) ---
    # O deck manda fixar o MLFLOW_TRACKING_URI por variável de ambiente. Aqui
    # ele vira config como qualquer outra: o default vale para todo mundo e
    # ``CHURN_TRACKING_URI`` sobrescreve (em time, aponta para o mlflow server).
    # sqlite e não ./mlruns porque o file store está em manutenção no MLflow 3.
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "churn"

    # --- Model Registry (Encontros 7 e 8) ---
    registered_model_name: str = "churn"
    champion_alias: str = "champion"
    # gate de promoção: abaixo disto o modelo não vira champion (slide 28)
    min_roc_auc: float = 0.80

    def hyperparams(self) -> dict[str, object]:
        """Só o que é hiperparâmetro — o que vai para ``log_params``.

        ``cfg.model_dump()`` inteiro também funciona, mas arrastaria caminhos
        de arquivo e a URI do tracking para dentro do run: valores que mudam
        de máquina para máquina e não explicam o resultado do experimento.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "test_size": self.test_size,
            "random_state": self.random_state,
        }


settings = Settings()
