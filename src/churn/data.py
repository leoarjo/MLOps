"""Carga, validação e limpeza dos dados.

Cada etapa é uma função pura: recebe um DataFrame, devolve outro, sem alterar
o original e sem efeito colateral. A orquestração fina fica em ``load_clean``.

A validação é feita em dois pontos (Encontro 4): o ``RawChurnSchema`` olha o
CSV como ele chega, o ``ChurnSchema`` confere o resultado da limpeza antes de
o dado virar feature. Se qualquer um dos contratos falhar, o pipeline para
antes do treino — barulhento e específico, em vez de silencioso.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.schema import ChurnSchema, RawChurnSchema


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


def load_clean(data_path: Path, id_column: str, target: str) -> pd.DataFrame:
    """Pipeline de dados: carrega -> valida cru -> limpa -> valida limpo.

    ``lazy=True`` faz o pandera coletar TODAS as violações antes de levantar
    ``SchemaErrors``, em vez de parar na primeira. O relatório sai com coluna,
    regra e valor — é o que torna a falha diagnosticável.

    O parâmetro ``target`` é mantido na assinatura por compatibilidade; a
    checagem do alvo agora vive declarada no schema.
    """
    df = load_raw(data_path)
    df = RawChurnSchema.validate(df, lazy=True)

    df = drop_identifier(df, id_column)
    df = coerce_total_charges(df)

    return ChurnSchema.validate(df, lazy=True)
