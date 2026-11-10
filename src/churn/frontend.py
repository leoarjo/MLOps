"""Front-end Streamlit da Churn API (Encontro 12).

Rodar (com a API de pé)::

    uv run streamlit run src/churn/frontend.py
    # http://localhost:8501

O Streamlit NÃO carrega o modelo: ele monta o registro completo do cliente,
manda para a API e mostra a resposta. Front e modelo ficam desacoplados — dá
para trocar o champion (ou o front) sem mexer no outro, e a API continua
sendo a única porta de entrada do modelo, inclusive para outros sistemas.

``CHURN_API_URL`` diz onde a API está (no compose: ``http://api:8000``).
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from churn.config import settings
from churn.schema import CONTRATOS, PAGAMENTOS

API_URL = settings.api_url.rstrip("/")
TIMEOUT = httpx.Timeout(30.0, connect=5.0)
CHUNK = 1_000  # registros por chamada no modo lote

SERVICOS_INTERNET = {
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de dispositivo",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
}


# ---------------------------------------------------------------- chamadas
def call(method: str, path: str, **kwargs: Any) -> httpx.Response:
    return httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)


@st.cache_data(ttl=30, show_spinner=False)
def model_info() -> dict[str, Any] | None:
    try:
        response = call("GET", "/model")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def show_validation_error(response: httpx.Response, offset: int = 0) -> None:
    """Traduz o 422 do FastAPI em 'linha X, campo Y: motivo'."""
    st.error("A API recusou o registro (contrato de entrada violado).")
    rows = []
    for err in response.json().get("detail", []):
        loc = [str(p) for p in err.get("loc", []) if p not in ("body", "records")]
        if loc and loc[0].isdigit():
            rows.append({"linha": int(loc[0]) + offset, "campo": ".".join(loc[1:]),
                         "problema": err.get("msg"), "valor": str(err.get("input"))})
        else:
            rows.append({"linha": None, "campo": ".".join(loc),
                         "problema": err.get("msg"), "valor": str(err.get("input"))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------- layout
st.set_page_config(page_title="Churn — previsão", page_icon="📉", layout="wide")

with st.sidebar:
    st.header("Modelo em produção")
    info = model_info()
    if info is None:
        st.error(f"API indisponível em `{API_URL}`")
        st.caption("Suba a API: `uv run uvicorn churn.api:app --port 8000`")
    else:
        st.success("API online")
        st.metric("Versão do champion", f"v{info.get('version', '?')}")
        st.metric("ROC AUC (teste)", info.get("roc_auc") or "—")
        st.caption(
            f"run `{str(info.get('run_id', ''))[:12]}` · "
            f"commit `{info.get('git_commit', '—')}` · "
            f"dado `{str(info.get('data_md5', '—'))[:8]}`"
        )
        st.caption(f"limiar de decisão: {info.get('decision_threshold', 0.5)}")
    st.divider()
    # texto, não link: no compose a URL é interna (http://api:8000) e não
    # abre no navegador de quem usa o front.
    st.caption(f"API: `{API_URL}`")

st.title("Previsão de churn")
st.caption(
    "Preencha o registro completo do cliente — o Streamlit envia para a API, "
    "que aplica o mesmo pré-processamento do treino e devolve a probabilidade."
)

aba_cliente, aba_lote = st.tabs(["Um cliente", "Lote (CSV)"])

# ---------------------------------------------------------------- um cliente
with aba_cliente:
    perfil, servicos, cobranca = st.columns(3, gap="large")

    with perfil:
        st.subheader("Perfil")
        customer_id = st.text_input("ID do cliente (opcional)", "7590-VHVEG")
        gender = st.radio("Gênero", ["Female", "Male"], horizontal=True,
                          format_func={"Female": "Feminino", "Male": "Masculino"}.get)
        senior = st.toggle("Idoso (SeniorCitizen)")
        partner = st.toggle("Tem cônjuge (Partner)", value=True)
        dependents = st.toggle("Tem dependentes")
        tenure = st.slider("Meses como cliente (tenure)", 0, 72, 1)

    with servicos:
        st.subheader("Serviços")
        phone = st.toggle("Telefone (PhoneService)")
        if phone:
            multiple = "Yes" if st.toggle("Múltiplas linhas") else "No"
        else:
            multiple = "No phone service"
        internet = st.selectbox(
            "Internet", ["DSL", "Fiber optic", "No"],
            format_func=lambda x: "Sem internet" if x == "No" else x,
        )
        extras: dict[str, str] = {}
        if internet == "No":
            st.caption("Sem internet: os serviços adicionais ficam como "
                       "*No internet service*.")
            extras = {col: "No internet service" for col in SERVICOS_INTERNET}
        else:
            escolhidos = st.multiselect(
                "Serviços adicionais", list(SERVICOS_INTERNET),
                default=["OnlineBackup"], format_func=SERVICOS_INTERNET.get,
            )
            extras = {col: "Yes" if col in escolhidos else "No"
                      for col in SERVICOS_INTERNET}

    with cobranca:
        st.subheader("Contrato e cobrança")
        contract = st.selectbox("Contrato", CONTRATOS)
        payment = st.selectbox("Forma de pagamento", PAGAMENTOS, index=2)
        paperless = st.toggle("Fatura digital (PaperlessBilling)", value=True)
        monthly = st.number_input("Mensalidade (MonthlyCharges)", 0.0, 200.0,
                                  29.85, step=0.5)
        auto_total = st.checkbox("TotalCharges = tenure × mensalidade", value=True)
        total = st.number_input(
            "Total gasto (TotalCharges)", 0.0, 20_000.0,
            round(tenure * monthly, 2) if auto_total else 29.85,
            step=10.0, disabled=auto_total,
        )
        if auto_total:
            total = round(tenure * monthly, 2)

    record = {
        "customerID": customer_id or None,
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "tenure": tenure,
        "PhoneService": "Yes" if phone else "No",
        "MultipleLines": multiple,
        "InternetService": internet,
        **extras,
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless else "No",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    st.divider()
    if st.button("Prever churn", type="primary", width="stretch"):
        try:
            response = call("POST", "/predict", json=record)
        except httpx.HTTPError as err:
            st.error(f"Não consegui falar com a API: {err}")
        else:
            if response.status_code == 422:
                show_validation_error(response)
            elif response.is_error:
                st.error(f"Erro {response.status_code}: {response.text}")
            else:
                result = response.json()
                proba = result["churn_probability"]
                alto = result["churn"] == "Yes"
                col_a, col_b = st.columns([1, 2])
                col_a.metric("Probabilidade de churn", f"{proba:.1%}")
                with col_b:
                    (st.error if alto else st.success)(
                        "**Alto risco de churn** — vale acionar retenção."
                        if alto else "**Baixo risco de churn.**"
                    )
                    st.progress(proba)

    with st.expander("Registro enviado para a API (JSON)"):
        st.code(json.dumps(record, indent=2, ensure_ascii=False), language="json")

# ---------------------------------------------------------------- lote
with aba_lote:
    st.write(
        "Envie um CSV com as mesmas colunas do `churn.csv`. A coluna `Churn`, "
        "se existir, é ignorada pelo modelo e usada só para comparar."
    )
    arquivo = st.file_uploader("CSV de clientes", type="csv")

    if arquivo is not None:
        frame = pd.read_csv(arquivo)
        # mesma limpeza do treino (data.coerce_total_charges): branco -> 0.0
        if "TotalCharges" in frame:
            frame["TotalCharges"] = pd.to_numeric(
                frame["TotalCharges"], errors="coerce"
            ).fillna(0.0)
        st.caption(f"{len(frame)} registros, {frame.shape[1]} colunas")
        st.dataframe(frame.head(), hide_index=True, width="stretch")

        if st.button("Prever o lote", type="primary"):
            records = json.loads(frame.to_json(orient="records"))
            predictions: list[dict[str, Any]] = []
            barra = st.progress(0.0, text="enviando para a API…")
            failed = False
            for start in range(0, len(records), CHUNK):
                chunk = records[start:start + CHUNK]
                try:
                    response = call("POST", "/predict/batch", json={"records": chunk})
                except httpx.HTTPError as err:
                    st.error(f"Não consegui falar com a API: {err}")
                    failed = True
                    break
                if response.status_code == 422:
                    show_validation_error(response, offset=start)
                    failed = True
                    break
                if response.is_error:
                    st.error(f"Erro {response.status_code}: {response.text}")
                    failed = True
                    break
                predictions.extend(response.json()["predictions"])
                barra.progress(min(1.0, (start + len(chunk)) / len(records)),
                               text=f"{start + len(chunk)}/{len(records)}")
            barra.empty()

            if not failed:
                out = frame.copy()
                out["churn_probability"] = [p["churn_probability"] for p in predictions]
                out["churn_previsto"] = [p["churn"] for p in predictions]

                c1, c2, c3 = st.columns(3)
                c1.metric("Clientes", len(out))
                c2.metric("Previstos como churn",
                          f"{(out['churn_previsto'] == 'Yes').mean():.1%}")
                if "Churn" in out:
                    c3.metric("Acerto vs. coluna Churn",
                              f"{(out['churn_previsto'] == out['Churn']).mean():.1%}")

                cols = [c for c in ("customerID", "churn_probability",
                                    "churn_previsto", "Churn") if c in out]
                st.dataframe(
                    out.sort_values("churn_probability", ascending=False)[cols],
                    hide_index=True, width="stretch",
                    column_config={"churn_probability": st.column_config.ProgressColumn(
                        "probabilidade", min_value=0.0, max_value=1.0, format="%.2f")},
                )
                st.download_button(
                    "Baixar resultado (CSV)", out.to_csv(index=False).encode(),
                    file_name="churn_previsoes.csv", mime="text/csv",
                )
