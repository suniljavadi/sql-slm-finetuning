from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("SQL_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="SQL Assistant", page_icon="SQL", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #152a24; --paper: #f3f6f0; --accent: #c64d32; --line: #cbd5cb; }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.45rem; font-weight: 800; padding-top: .5rem; }
    code, pre { font-family: 'DM Mono', monospace !important; }
    div.stButton > button[kind="primary"] { background: var(--accent); border: 0; color: white; }
    div.stButton > button[kind="primary"]:hover { background: #a9402a; color: white; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("AMD ROCm  /  QLoRA")
st.title("SQL Assistant")
st.markdown("Generate a read-only query against your schema.")
st.divider()

with st.form("sql_request"):
    question = st.text_area(
        "Question",
        placeholder="List customer names in Boston",
        height=110,
        max_chars=4000,
    )
    schema = st.text_area(
        "Database schema",
        placeholder="customers(id, name, city)",
        height=120,
        max_chars=4000,
    )
    submitted = st.form_submit_button("Generate SQL", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.error("Enter a question to generate a query.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/generate",
                json={"prompt": question, "schema": schema},
                timeout=(5, 120),
            )
            response.raise_for_status()
            sql = response.json()["sql"]
            st.session_state["sql_result"] = sql
        except requests.RequestException as exc:
            st.error(f"The model service is unavailable: {exc}")
        except (KeyError, ValueError) as exc:
            st.error(f"The model service returned an invalid response: {exc}")

sql_result = st.session_state.get("sql_result")
if sql_result:
    st.divider()
    st.subheader("Generated query")
    st.code(sql_result, language="sql")
    st.download_button(
        "Download SQL",
        data=sql_result,
        file_name="query.sql",
        mime="text/sql",
    )