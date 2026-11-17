"""Exporta o champion do registry para uma pasta servível (Encontro 12).

Entrypoint::

    python -m churn.export            # -> artifacts/champion/

O registry responde "qual versão é produção" (``models:/churn@champion``),
mas vive no ``mlflow.db`` + ``mlruns/`` da máquina de quem treinou. A API no
EC2 não deve depender disso: ela recebe uma pasta autocontida, que entra na
imagem Docker junto com o código.

    artifacts/champion/
    ├─ model/          # o modelo MLflow (flavor sklearn, com signature)
    ├─ encoder.pkl     # o CategoricalEncoder ajustado no treino daquele run
    └─ metadata.json   # versão, run, roc_auc, data_md5, git_commit

Promover um modelo novo = ``churn.registry`` -> ``churn.export`` -> rebuild da
imagem. A imagem passa a ser o artefato imutável: a tag dela diz exatamente
qual versão do modelo está servindo.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from churn.config import Settings, settings
from churn.registry import champion_uri
from churn.serving import MODEL_SUBDIR, ModelBundle, write_sidecars

logger = logging.getLogger(__name__)

ENCODER_ARTIFACT = "encoder/encoder.pkl"


def export_champion(cfg: Settings = settings, out_dir: Path | None = None) -> Path:
    """Baixa o champion + o encoder do run dele e grava o bundle."""
    out_dir = out_dir or cfg.model_dir
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = MlflowClient(tracking_uri=cfg.tracking_uri)

    version = client.get_model_version_by_alias(
        cfg.registered_model_name, cfg.champion_alias
    )

    # O encoder é artefato do RUN (experiments.log_model_with_contract), não
    # do modelo registrado. Um champion vindo de um run sem encoder — o
    # ``churn.model`` não loga — não é servível, e isso tem que parar aqui,
    # antes de mexer no bundle que já está em artifacts/champion.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path=ENCODER_ARTIFACT, dst_path=tmp
            )
        except Exception as err:  # o MLflow levanta tipos diferentes por backend
            raise FileNotFoundError(
                f"o run {version.run_id} (champion v{version.version}) nao tem "
                f"'{ENCODER_ARTIFACT}'. Promova um run do grid "
                "(`python -m churn.experiments`), que loga o encoder."
            ) from err
        encoder = pickle.loads(Path(local).read_bytes())

    metadata = {
        "registered_model": cfg.registered_model_name,
        "alias": cfg.champion_alias,
        "version": str(version.version),
        "run_id": version.run_id,
        "roc_auc": version.tags.get("roc_auc"),
        "data_md5": version.tags.get("data_md5"),
        "git_commit": version.tags.get("git_commit"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # grava numa pasta temporária e só então troca: um export que quebra no
    # meio não deixa um bundle pela metade no lugar do anterior.
    staging = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    # O pacote MLflow é baixado como foi registrado — MLmodel, signature,
    # input_example, requirements — em vez de recarregado e salvo de novo,
    # o que perderia o contrato que o Encontro 8 colocou ali.
    mlflow.artifacts.download_artifacts(
        artifact_uri=champion_uri(cfg), dst_path=str(staging / MODEL_SUBDIR)
    )
    write_sidecars(staging, encoder, metadata)

    # confere que o bundle abre ANTES de substituir o anterior
    ModelBundle.load(staging)
    shutil.rmtree(out_dir, ignore_errors=True)
    staging.rename(out_dir)

    logger.info("champion v%s exportado para %s", version.version, out_dir)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Exporta o champion para servir.")
    parser.add_argument("--out", type=Path, default=settings.model_dir)
    args = parser.parse_args()

    out = export_champion(out_dir=args.out)
    print(f"{champion_uri()} exportado para {out}/")


if __name__ == "__main__":
    main()
