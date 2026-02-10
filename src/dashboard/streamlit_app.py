import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Vietnam Air Quality", page_icon="🇻🇳")

@st.cache_data
def load_data():
    # Essaie de charger le CSV local
    possible_paths = ["data/aqi_data.csv", "src/data/aqi_data.csv", "aqi_data.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Nettoyage date basique
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            return df
    return None

def init_db():
    try: return st.connection("postgresql", type="sql")
    except: return None

# --- APP ---
st.title("🇻🇳 Qualité de l'Air (Version Stable)")

# 1. LIVE DATA
st.subheader("🔴 En Direct (Base de Données)")
try:
    conn = init_db()
    if conn:
        df_live = conn.query('SELECT * FROM aqi_data ORDER BY timestamp DESC LIMIT 50;', ttl="10m")
        st.dataframe(df_live)
    else:
        st.info("Base de données en attente.")
except:
    st.warning("Mode hors-ligne (BDD non connectée)")

# 2. ARCHIVES
st.subheader("📊 Historique (CSV)")
df = load_data()
if df is not None:
    cities = df['location'].unique()
    city = st.selectbox("Ville", cities)
    subset = df[df['location'] == city]
    st.line_chart(subset, x='timestamp', y='aqi')
else:
    st.error("CSV introuvable.")
