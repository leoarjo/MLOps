"""Contrato de dados do projeto churn (pandera).

Se o Pydantic define o contrato da configuração (``config.py``), o pandera
define o contrato do DataFrame. Mesmo paradigma: uma classe declarativa,
versionada junto do código e revisável em PR.

Há dois schemas, aplicados em pontos distintos do pipeline:

``RawChurnSchema``
    valida o CSV como ele chega — antes de qualquer limpeza. É o único
    momento em que ``customerID`` ainda existe, então a checagem de
    unicidade mora aqui.

``ChurnSchema``
    valida o DataFrame já limpo, imediatamente antes de virar features.
    Tipos, faixas e categorias.

Validar nos dois pontos é o que impede que ``coerce=True`` mascare sujeira
real: o schema cru vê o dado como ele é, o schema limpo confere o resultado
da limpeza.

As faixas abaixo saíram do profiling do nosso próprio ``churn.csv`` (7043
linhas): tenure 0–72, MonthlyCharges 18.25–118.75, TotalCharges 0–8269.27,
11 brancos em TotalCharges, todos em clientes com tenure 0.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

# --- categorias observadas no dataset ---
SIM_NAO = ["Yes", "No"]
SIM_NAO_SEM_TELEFONE = ["Yes", "No", "No phone service"]
SIM_NAO_SEM_INTERNET = ["Yes", "No", "No internet service"]
CONTRATOS = ["Month-to-month", "One year", "Two year"]
PAGAMENTOS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]


class RawChurnSchema(pa.DataFrameModel):
    """Contrato do CSV cru, antes da limpeza."""

    customerID: Series[str] = pa.Field(unique=True, nullable=False)
    tenure: Series[int] = pa.Field(ge=0, le=72)
    Churn: Series[str] = pa.Field(isin=SIM_NAO, nullable=False)

    class Config:
        coerce = False   # o dado cru é olhado como está, sem conversão
        strict = False


class ChurnSchema(pa.DataFrameModel):
    """Contrato do DataFrame limpo, antes de virar features."""

    # --- demografia ---
    gender: Series[str] = pa.Field(isin=["Female", "Male"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=SIM_NAO)
    Dependents: Series[str] = pa.Field(isin=SIM_NAO)

    # --- contrato e serviços ---
    tenure: Series[int] = pa.Field(ge=0, le=72)
    PhoneService: Series[str] = pa.Field(isin=SIM_NAO)
    MultipleLines: Series[str] = pa.Field(isin=SIM_NAO_SEM_TELEFONE)
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(isin=SIM_NAO_SEM_INTERNET)
    OnlineBackup: Series[str] = pa.Field(isin=SIM_NAO_SEM_INTERNET)
    DeviceProtection: Series[str] = pa.Field(isin=SIM_NAO_SEM_INTERNET)
    TechSupport: Series[str] = pa.Field(isin=SIM_NAO_SEM_INTERNET)
    StreamingTV: Series[str] = pa.Field(isin=SIM_NAO_SEM_INTERNET)
    StreamingMovies: Series[str] = pa.Field(isin=SIM_NAO_SEM_INTERNET)
    Contract: Series[str] = pa.Field(isin=CONTRATOS)
    PaperlessBilling: Series[str] = pa.Field(isin=SIM_NAO)
    PaymentMethod: Series[str] = pa.Field(isin=PAGAMENTOS)

    # --- valores ---
    # Faixas propositalmente folgadas em relação ao observado (18.25–118.75 e
    # 0–8269.27). O slide 15 é explícito: comece frouxo e aperte com o tempo,
    # olhando os failure_cases. Um schema que barra dado legítimo vira ruído
    # e acaba desligado.
    MonthlyCharges: Series[float] = pa.Field(ge=0, le=200)
    TotalCharges: Series[float] = pa.Field(ge=0, le=20_000, nullable=False)

    # --- alvo ---
    Churn: Series[str] = pa.Field(isin=SIM_NAO, nullable=False)

    class Config:
        coerce = True    # converte tipos ao validar (TotalCharges texto -> float)
        strict = False   # tolera colunas extras (ex.: features derivadas)
