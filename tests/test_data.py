import pandera.errors as pae
import pytest

from churn.data import coerce_total_charges, drop_identifier, load_clean


def test_coerce_total_charges_converte_branco_em_zero(raw_frame):
    out = coerce_total_charges(raw_frame)
    assert out["TotalCharges"].dtype.kind == "f"
    # a linha com tenure 0 tinha TotalCharges em branco -> 0.0
    assert out.loc[raw_frame["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0


def test_coerce_e_funcao_pura(raw_frame):
    _ = coerce_total_charges(raw_frame)
    assert raw_frame["TotalCharges"].dtype.kind != "f"  # original intacto


def test_drop_identifier(raw_frame):
    out = drop_identifier(raw_frame, "customerID")
    assert "customerID" not in out.columns


def test_load_clean_aceita_dado_valido(tmp_path, raw_frame):
    csv = tmp_path / "churn.csv"
    raw_frame.to_csv(csv, index=False)
    out = load_clean(csv, "customerID", "Churn")
    assert "customerID" not in out.columns
    assert out["TotalCharges"].dtype.kind == "f"


def test_load_clean_para_o_pipeline_com_dado_sujo(tmp_path, raw_frame):
    """A dor do slide 3: dado sujo agora falha alto, antes do treino."""
    sujo = raw_frame.copy()
    sujo.loc[0, "tenure"] = 999
    csv = tmp_path / "sujo.csv"
    sujo.to_csv(csv, index=False)

    with pytest.raises(pae.SchemaErrors):
        load_clean(csv, "customerID", "Churn")
