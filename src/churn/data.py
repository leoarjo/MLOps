"""Carga, validação e limpeza dos dados.

Cada etapa é uma função pura: recebe um DataFrame, devolve outro, sem alterar
o original e sem efeito colateral. A orquestração fina fica em ``load_clean``.

A validação é feita em dois pontos (Encontro 4): o ``RawChurnSchema`` olha o
CSV como ele chega, o ``ChurnSchema`` confere o resultado da limpeza antes de
o dado virar feature. Se qualquer um dos contratos falhar, o pipeline para
antes do treino — barulhento e específico, em vez de silencioso.

Quando um contrato falha, os ``failure_cases`` (coluna, regra, valor) vão
para o log antes de a exceção subir — é o "logar os failure_cases" do
slide 11: quem investiga o incidente não precisa reproduzir o erro.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera.errors as pae
import pandera.pandas as pa

from churn.schema import ChurnSchema, RawChurnSchema

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Lê o CSV cru do disco."""
    return pd.read_csv(path)


def drop_identifier(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Remove a coluna de identificador — é chave, não feature."""
    return df.drop(columns=[id_column])


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte ``TotalCharges`` (texto, com brancos) em número.

    Os brancos correspondem a clientes com ``tenure == 0`` (acabaram de
    entrar), cujo total gasto é de fato 0 — e não 2200, o número mágico do
    script original.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    return out


def validate(schema: type[pa.DataFrameModel], df: pd.DataFrame) -> pd.DataFrame:
    """Aplica um contrato; se falhar, loga cada violação e relança o erro."""
    try:
        return schema.validate(df, lazy=True)
    except pae.SchemaErrors as err:
        cases = err.failure_cases
        logger.error(
            "%s: %d violação(ões) do contrato de dados\n%s",
            schema.__name__,
            len(cases),
            cases[["column", "check", "failure_case", "index"]].to_string(index=False),
        )
        raise


def load_clean(data_path: Path, id_column: str, target: str) -> pd.DataFrame:
    """Pipeline de dados: carrega -> valida cru -> limpa -> valida limpo.

    ``lazy=True`` faz o pandera coletar TODAS as violações antes de levantar
    ``SchemaErrors``, em vez de parar na primeira. O relatório sai com coluna,
    regra e valor — é o que torna a falha diagnosticável.

    O parâmetro ``target`` é mantido na assinatura por compatibilidade; a
    checagem do alvo agora vive declarada no schema.
    """
    df = load_raw(data_path)
    df = validate(RawChurnSchema, df)

    df = drop_identifier(df, id_column)
    df = coerce_total_charges(df)

    return validate(ChurnSchema, df)
