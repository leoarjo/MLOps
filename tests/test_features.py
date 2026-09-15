from churn.data import coerce_total_charges, drop_identifier
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)


def _limpo(raw_frame):
    return coerce_total_charges(drop_identifier(raw_frame, "customerID"))


def test_add_derived_features_cria_gasto_por_mes(raw_frame):
    out = add_derived_features(_limpo(raw_frame))
    assert "gasto_por_mes" in out.columns
    # tenure 0 não pode gerar divisão por zero
    assert out["gasto_por_mes"].notna().all()


def test_split_features_target_binariza_o_alvo(raw_frame):
    features, labels = split_features_target(
        _limpo(raw_frame), target="Churn", positive_label="Yes"
    )
    assert "Churn" not in features.columns
    assert set(labels.unique()) <= {0, 1}
    assert labels.tolist() == [0, 1, 0, 1]


def test_encoder_nao_vaza_categoria_do_teste(raw_frame):
    """Categoria inédita no teste vira -1, não altera o mapeamento do treino."""
    frame = _limpo(raw_frame)
    features, _ = split_features_target(frame, "Churn", "Yes")

    treino = features.iloc[:2]
    teste = features.iloc[2:].copy()
    teste.loc[:, "Contract"] = "Vitalício"  # categoria que o treino nunca viu

    encoder = CategoricalEncoder().fit(treino)
    assert (encoder.transform(teste)["Contract"] == -1).all()
