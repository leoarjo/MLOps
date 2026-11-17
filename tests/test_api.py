"""Testes da API (Encontro 12).

Nada de mlflow.db nem do champion real: cada teste monta um bundle minúsculo
a partir da mini-amostra do ``conftest.py``, pelo MESMO caminho do treino
(limpeza -> features -> encoder). É o que deixa o teste rápido, determinístico
e independente de quem rodou o grid na máquina.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from churn.api import create_app  # noqa: E402
from churn.data import coerce_total_charges, drop_identifier  # noqa: E402
from churn.features import (  # noqa: E402
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)
from churn.serving import CustomerRecord, ModelBundle  # noqa: E402


@pytest.fixture
def bundle_dir(raw_frame, tmp_path):
    frame = add_derived_features(
        coerce_total_charges(drop_identifier(raw_frame, "customerID"))
    )
    features, labels = split_features_target(frame, "Churn", "Yes")
    encoder = CategoricalEncoder().fit(features)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        encoder.transform(features), labels
    )
    path = tmp_path / "champion"
    ModelBundle(model, encoder, {"version": "7", "roc_auc": "0.9"}).save(path)
    return path


@pytest.fixture
def client(bundle_dir):
    with TestClient(create_app(bundle_dir)) as c:  # o `with` dispara o lifespan
        yield c


@pytest.fixture
def records(raw_frame):
    clean = coerce_total_charges(raw_frame).drop(columns=["Churn"])
    return clean.to_dict(orient="records")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_mostra_a_linhagem_do_bundle(client):
    info = client.get("/model").json()
    assert info["version"] == "7"
    assert info["decision_threshold"] == 0.5


def test_predict_devolve_probabilidade_e_rotulo(client, records):
    body = client.post("/predict", json=records[0]).json()
    assert body["customerID"] == "A1"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn"] == ("Yes" if body["churn_probability"] >= 0.5 else "No")


def test_batch_preserva_ordem_e_bate_com_o_modelo(client, bundle_dir, records):
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert [p["customerID"] for p in preds] == ["A1", "A2", "A3", "A4"]

    # paridade treino x serving: a API não pode transformar diferente
    bundle = ModelBundle.load(bundle_dir)
    esperado = bundle.predict_proba([CustomerRecord(**r) for r in records])
    obtido = [p["churn_probability"] for p in preds]
    assert obtido == pytest.approx(esperado, abs=1e-6)


def test_coluna_extra_churn_e_ignorada(client, records):
    response = client.post("/predict", json={**records[0], "Churn": "Yes"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("campo", "valor"),
    [("gender", "X"), ("tenure", 999), ("Contract", "Monthly"), ("TotalCharges", -1)],
)
def test_registro_fora_do_contrato_vira_422(client, records, campo, valor):
    response = client.post("/predict", json={**records[0], campo: valor})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", campo]


def test_campo_faltando_vira_422(client, records):
    incompleto = {k: v for k, v in records[0].items() if k != "PaymentMethod"}
    assert client.post("/predict", json=incompleto).status_code == 422


def test_batch_vazio_vira_422(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_limiar_configuravel(bundle_dir, records):
    with TestClient(create_app(bundle_dir, threshold=0.0)) as c:
        preds = c.post("/predict/batch", json={"records": records}).json()
    assert {p["churn"] for p in preds["predictions"]} == {"Yes"}


def test_api_nao_sobe_sem_bundle(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn.export"):
        with TestClient(create_app(tmp_path / "nao-existe")):
            pass
