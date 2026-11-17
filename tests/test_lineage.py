from pathlib import Path

from churn import lineage


def test_data_md5_le_o_ponteiro_do_dvc(tmp_path):
    pointer = tmp_path / "churn.csv.dvc"
    pointer.write_text(
        "outs:\n- md5: d390bd07b5514a2256a2396993b8b0e3\n  size: 981685\n"
        "  hash: md5\n  path: churn.csv\n"
    )
    assert lineage.data_md5(pointer) == "d390bd07b5514a2256a2396993b8b0e3"


def test_data_md5_nao_quebra_sem_ponteiro(tmp_path):
    """Dentro do container não há .dvc — a tag some, o treino continua."""
    assert lineage.data_md5(tmp_path / "nao-existe.dvc") == lineage.UNKNOWN


def test_data_md5_nao_quebra_com_yaml_invalido(tmp_path):
    pointer = tmp_path / "quebrado.dvc"
    pointer.write_text("isto: nao: e: um: ponteiro\n")
    assert lineage.data_md5(pointer) == lineage.UNKNOWN


def test_tags_tem_as_tres_chaves():
    assert set(lineage.tags(Path("data/churn.csv.dvc"))) == {
        "data_md5",
        "git_commit",
        "git_dirty",
    }
