"""Inferência do champion — o que a API (e qualquer outro consumidor) usa.

Três peças, nesta ordem:

``CustomerRecord``
    o contrato de ENTRADA: um registro de cliente completo, como uma linha do
    ``churn.csv`` sem o alvo. É Pydantic (e não pandera) porque quem fala com
    a API é JSON, não DataFrame: um campo errado vira um 422 que aponta o
    campo e o valor, antes de o modelo ser tocado.

``ModelBundle``
    o champion exportado por ``python -m churn.export``: o classificador, o
    encoder ajustado no treino e os metadados de linhagem (versão, run,
    roc_auc, md5 do dado, commit). Carregado UMA vez, na subida da API.

``ModelBundle.predict``
    aplica exatamente a mesma transformação do treino — ``TotalCharges``
    numérico, ``gasto_por_mes``, encoder, mesma ordem de colunas — e devolve
    a probabilidade de churn. É aqui que se evita o *training-serving skew*:
    as funções são as mesmas de ``features.py``, não uma cópia.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from churn.features import CategoricalEncoder, add_derived_features
from churn.schema import (
    CONTRATOS,
    PAGAMENTOS,
    SIM_NAO,
    SIM_NAO_SEM_INTERNET,
    SIM_NAO_SEM_TELEFONE,
)

logger = logging.getLogger(__name__)

MODEL_SUBDIR = "model"
ENCODER_FILE = "encoder.pkl"
METADATA_FILE = "metadata.json"

# As categorias vêm do schema.py: o contrato da API e o contrato do dado de
# treino não podem divergir. Se uma categoria nova entrar no schema, a API
# passa a aceitá-la sem ninguém lembrar de editar este arquivo.
SimNao = Literal[tuple(SIM_NAO)]  # type: ignore[valid-type]
SimNaoTelefone = Literal[tuple(SIM_NAO_SEM_TELEFONE)]  # type: ignore[valid-type]
SimNaoInternet = Literal[tuple(SIM_NAO_SEM_INTERNET)]  # type: ignore[valid-type]
Contrato = Literal[tuple(CONTRATOS)]  # type: ignore[valid-type]
Pagamento = Literal[tuple(PAGAMENTOS)]  # type: ignore[valid-type]


class CustomerRecord(BaseModel):
    """Um cliente — todas as colunas do ``churn.csv`` menos o ``Churn``.

    As faixas repetem as do ``ChurnSchema`` (o mesmo "comece frouxo" do
    Encontro 4). ``customerID`` é opcional: não é feature, só volta na
    resposta para o chamador casar a previsão com o registro dele. Campos
    extras (ex.: a própria coluna ``Churn`` de um CSV) são ignorados.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        },
    )

    customerID: str | None = None

    # --- demografia ---
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: SimNao
    Dependents: SimNao

    # --- contrato e serviços ---
    tenure: int = Field(ge=0, le=72, description="meses como cliente")
    PhoneService: SimNao
    MultipleLines: SimNaoTelefone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: SimNaoInternet
    OnlineBackup: SimNaoInternet
    DeviceProtection: SimNaoInternet
    TechSupport: SimNaoInternet
    StreamingTV: SimNaoInternet
    StreamingMovies: SimNaoInternet
    Contract: Contrato
    PaperlessBilling: SimNao
    PaymentMethod: Pagamento

    # --- valores ---
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(
        ge=0, le=20_000, description="total gasto; 0 para cliente novo (tenure 0)"
    )


class Prediction(BaseModel):
    """A resposta para um cliente."""

    customerID: str | None
    churn_probability: float = Field(ge=0, le=1)
    churn: Literal["Yes", "No"]


def write_sidecars(
    model_dir: Path, encoder: CategoricalEncoder, metadata: dict[str, Any]
) -> None:
    """Grava o que acompanha o modelo MLflow: encoder e metadados."""
    (model_dir / ENCODER_FILE).write_bytes(pickle.dumps(encoder))
    (model_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )


@dataclass
class ModelBundle:
    """Champion pronto para servir: modelo + encoder + linhagem."""

    model: Any
    encoder: CategoricalEncoder
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, model_dir: Path) -> "ModelBundle":
        """Lê o bundle de ``model_dir`` — falha alto se faltar alguma peça."""
        import mlflow.sklearn  # import tardio: só quem serve paga o custo

        model_path = model_dir / MODEL_SUBDIR
        encoder_path = model_dir / ENCODER_FILE
        missing = [p for p in (model_path, encoder_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"bundle do modelo incompleto em '{model_dir}' (faltando: "
                f"{', '.join(str(p) for p in missing)}). "
                "Rode `uv run python -m churn.export` antes de subir a API."
            )

        # flavor sklearn (e não pyfunc) porque a API devolve PROBABILIDADE:
        # o pyfunc só expõe predict(), que já vem cortado em 0/1.
        model = mlflow.sklearn.load_model(str(model_path))
        encoder = pickle.loads(encoder_path.read_bytes())

        metadata_path = model_dir / METADATA_FILE
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        logger.info("bundle carregado de %s: %s", model_dir, metadata)
        return cls(model=model, encoder=encoder, metadata=metadata)

    def save(self, model_dir: Path) -> None:
        """Grava o bundle a partir de objetos em memória (usado nos testes).

        O ``churn.export`` não passa por aqui para o modelo: ele baixa o
        pacote MLflow do registry intacto e só grava os arquivos ao lado.
        """
        import mlflow.sklearn

        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(
            self.model,
            str(model_dir / MODEL_SUBDIR),
            serialization_format="cloudpickle",
        )
        write_sidecars(model_dir, self.encoder, self.metadata)

    # ------------------------------------------------------------ predição
    def features(self, records: list[CustomerRecord]) -> pd.DataFrame:
        """Registros -> matriz no formato exato que o modelo viu no treino."""
        frame = pd.DataFrame(
            [r.model_dump(exclude={"customerID"}) for r in records]
        )
        frame = add_derived_features(frame)
        encoded = self.encoder.transform(frame)
        # o sklearn confere nome E ordem das colunas contra o treino
        columns = list(getattr(self.model, "feature_names_in_", encoded.columns))
        return encoded[columns]

    def predict_proba(self, records: list[CustomerRecord]) -> list[float]:
        probabilities = self.model.predict_proba(self.features(records))[:, 1]
        return [float(p) for p in probabilities]

    def predict(
        self, records: list[CustomerRecord], threshold: float = 0.5
    ) -> list[Prediction]:
        return [
            Prediction(
                customerID=record.customerID,
                churn_probability=round(proba, 6),
                churn="Yes" if proba >= threshold else "No",
            )
            for record, proba in zip(records, self.predict_proba(records))
        ]
