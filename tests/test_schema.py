import pandas as pd
import pandera.errors as pae
import pytest

from churn.schema import ChurnSchema, RawChurnSchema


@pytest.fixture
def limpo(raw_frame):
    """A fixture crua, já com TotalCharges numérico e sem customerID."""
    out = raw_frame.drop(columns=["customerID"]).copy()
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    return out


def test_dado_valido_passa(limpo):
    validado = ChurnSchema.validate(limpo, lazy=True)
    assert len(validado) == len(limpo)


def test_tenure_fora_da_faixa_falha(limpo):
    quebrado = limpo.copy()
    quebrado.loc[0, "tenure"] = 999
    with pytest.raises(pae.SchemaErrors) as exc:
        ChurnSchema.validate(quebrado, lazy=True)
    assert "tenure" in exc.value.failure_cases["column"].values


def test_categoria_desconhecida_falha(limpo):
    quebrado = limpo.copy()
    quebrado.loc[0, "Contract"] = "Vitalício"
    with pytest.raises(pae.SchemaErrors) as exc:
        ChurnSchema.validate(quebrado, lazy=True)
    assert "Contract" in exc.value.failure_cases["column"].values


def test_lazy_coleta_todas_as_falhas(limpo):
    """Com lazy=True o relatório traz as duas violações, não só a primeira."""
    quebrado = limpo.copy()
    quebrado.loc[0, "tenure"] = 999
    quebrado.loc[1, "MonthlyCharges"] = -30.0
    with pytest.raises(pae.SchemaErrors) as exc:
        ChurnSchema.validate(quebrado, lazy=True)
    colunas = set(exc.value.failure_cases["column"])
    assert {"tenure", "MonthlyCharges"} <= colunas


def test_id_duplicado_falha_no_schema_cru(raw_frame):
    quebrado = raw_frame.copy()
    quebrado.loc[1, "customerID"] = quebrado.loc[0, "customerID"]
    with pytest.raises(pae.SchemaErrors) as exc:
        RawChurnSchema.validate(quebrado, lazy=True)
    assert "customerID" in exc.value.failure_cases["column"].values
