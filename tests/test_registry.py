import pandas as pd
import pytest

from churn import registry
from churn.config import Settings


@pytest.fixture
def runs() -> pd.DataFrame:
    """Três runs empatados na prática e um claramente pior."""
    return pd.DataFrame(
        {
            "run_id": ["r500", "r200", "r100", "r_ruim"],
            "metrics.roc_auc": [0.822032, 0.822031, 0.821541, 0.790000],
            "params.n_estimators": ["500", "200", "100", "500"],
            "params.max_depth": ["8", "8", "8", "None"],
            "tags.data_md5": ["ca5c74c9", "ca5c74c9", "ca5c74c9", "ca5c74c9"],
        }
    )


def test_gate_aprova_acima_do_limiar():
    assert registry.passes_gate({"roc_auc": 0.8215}, 0.80)


def test_gate_reprova_abaixo_do_limiar():
    assert not registry.passes_gate({"roc_auc": 0.75}, 0.80)


def test_gate_reprova_sem_a_metrica():
    """Métrica ausente não pode ser lida como aprovação."""
    assert not registry.passes_gate({"accuracy": 0.99}, 0.80)


def test_best_run_desempata_pelo_modelo_mais_barato(monkeypatch, runs):
    monkeypatch.setattr(registry, "search_finished_runs", lambda cfg: runs)
    assert registry.best_run(Settings())["run_id"] == "r100"


def test_best_run_ignora_quem_esta_fora_do_empate(monkeypatch, runs):
    """Ser o mais barato não basta: fora do empate, não concorre."""
    runs.loc[runs["run_id"] == "r_ruim", "params.n_estimators"] = "10"
    monkeypatch.setattr(registry, "search_finished_runs", lambda cfg: runs)
    assert registry.best_run(Settings())["run_id"] == "r100"


def test_best_run_sem_runs_falha_com_instrucao(monkeypatch):
    monkeypatch.setattr(registry, "search_finished_runs", lambda cfg: pd.DataFrame())
    with pytest.raises(ValueError, match="churn.experiments"):
        registry.best_run(Settings())


def test_champion_uri():
    assert registry.champion_uri(Settings()) == "models:/churn@champion"
