import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Model Overview — Downturn Risk Monitor",
    page_icon="📖",
    layout="wide",
)

docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "model_explainer.html")

if not os.path.exists(docs_path):
    st.error(f"Model overview document not found at `{docs_path}`.")
else:
    with open(docs_path, encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=1350, scrolling=True)
