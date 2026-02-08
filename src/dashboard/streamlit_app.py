import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Vietnam Air Quality AI", page_icon="🇻🇳", layout="wide")

# --- 2. FONCTIONS DE CHARGEMENT ---

@st.cache_data
def load_archive_data():
    """
    Charge le CSV, convertit en HEURE VIETNAM (UTC+7) et nettoie les données.
    """
    # Recherche intelligente du fichier
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "data/aqi_data.csv",
        "src/data/aqi_data.csv",
        os.path.join(current_dir, "aqi_data.csv"),
        "aqi_data.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except:
                continue
    
    if df is not None:
        # --- GESTION TEMPORELLE VIETNAM ---
        # 1. Lire en UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        # 2. Convertir en heure locale (Vietnam)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh')
        # 3. Retirer le fuseau pour l'affichage (rend la date "naïve")
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        # Nettoyage colonnes numériques
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna().sort_values('timestamp')
    else:
        return None

def init_connection():
    try:
        return st.connection("postgresql", type="sql")
    except:
        return None

def load_live_data(conn):
    try:
        # On charge les données fraîches
        return conn.query('SELECT * FROM aqi_data ORDER BY timestamp DESC LIMIT 500;', ttl="10m")
    except:
        return pd.DataFrame()

def get_aqi_color(aqi):
    if aqi <= 50: return "#00E400"
    elif aqi <= 100: return "#FFFF00"
    elif aqi <= 150: return "#FF7E00"
    elif aqi <= 200: return "#FF0000"
    elif aqi <= 300: return "#8F3F97"
    else: return "#7E0023"

# --- 3. INTERFACE ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["📊 Archives & Performance IA", "🔴 Live Data (Temps Réel)"])
st.sidebar.markdown("---")
st.sidebar.info("Projet : Vietnam Air Quality Forecasting")

# ==============================================================================
# ONGLET 1 : ARCHIVES & IA
# ==============================================================================
if page == "📊 Archives & Performance IA":
    st.title("🧠 Performance du Modèle (Archives)")
    st.markdown("Analyse basée sur l'historique CSV (Heure Locale Vietnam).")
    
    df = load_archive_data()

    if df is not None:
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            locations = sorted(df['location'].unique())
            selected_location = st.selectbox("📍 Ville", locations)
            df_loc = df[df['location'] == selected_location]
        
        with col2:
            min_date = df_loc['timestamp'].min().date()
            max_date = df_loc['timestamp'].max().date()
            test_date = st.date_input("📅 Date à prédire", max_date, min_value=min_date, max_value=max_date)

        # --- DÉCOUPE DES DONNÉES (Fix "Pas assez de données") ---
        split_ts = pd.Timestamp(test_date)
        
        # 1. Tentative Standard (Jours d'avant vs Jour J)
        train_df = df_loc[df_loc['timestamp'] < split_ts]
        test_df = df_loc[df_loc['timestamp'].dt.date == test_date]
        
        # 2. Mode Secours (Si fichier ne contient qu'un seul jour)
        if len(train_df) < 10:
            # On coupe la journée en deux : 80% matin (Train) / 20% soir (Test)
            day_data = df_loc[df_loc['timestamp'].dt.date == test_date].sort_values('timestamp')
            if len(day_data) > 5:
                split_idx = int(len(day_data) * 0.8)
                train_df = day_data.iloc[:split_idx]
                test_df = day_data.iloc[split_idx:]
                st.warning("⚠️ Mode 'Données limitées' : Entraînement sur le début de journée, test sur la fin.")

        # --- IA & RÉSULTATS ---
        if len(train_df) > 5 and len(test_df) > 0:
            
            features = ['pm25', 'no2', 'so2', 'co', 'o3']
            features = [f for f in features if f in df.columns]
            
            X_train = train_df[features]
            y_train = train_df['aqi']
            X_test = test_df[features]
            y_test = test_df['aqi']

            # Random Forest
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Lignes d'entraînement", len(train_df))
            kpi2.metric("Précision (R²)", f"{r2:.2f}")
            kpi3.metric("Erreur (MAE)", f"{mae:.1f}")

            # Graphique
            st.subheader(f"📉 Évolution à {selected_location}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df['timestamp'], y=y_test, mode='lines+markers', name='Réalité'))
            fig.add_trace(go.Scatter(x=test_df['timestamp'], y=y_pred, mode='lines', name='Prédiction IA', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("❌ Pas assez de données pour cette date et ce lieu.")

    else:
        st.error("Fichier CSV introuvable (Vérifiez le dossier 'data').")

# ==============================================================================
# ONGLET 2 : LIVE DATA
# ==============================================================================
elif page == "🔴 Live Data (Temps Réel)":
    st.title("📡 Monitoring NeonDB")
    conn = init_connection()
    
    if conn:
        df_live = load_live_data(conn)
        if not df_live.empty:
            # Conversion UTC -> Vietnam pour l'affichage Live aussi
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], utc=True)
            df_live['timestamp'] = df_live['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
            
            latest_time = df_live['timestamp'].max()
            st.success(f"Dernière mise à jour (Heure Vietnam) : {latest_time}")
            
            # KPIs
            latest_data = df_live.sort_values('timestamp', ascending=False).drop_duplicates('location').head(4)
            cols = st.columns(len(latest_data))
            for idx, row in enumerate(latest_data.itertuples()):
                with cols[idx]:
                    try:
                        color = get_aqi_color(row.aqi)
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 10px; text-align: center;">
                            <h3>{row.location}</h3>
                            <h1>{int(row.aqi)}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    except: pass
            
            st.dataframe(df_live, use_container_width=True)
            if st.button("Actualiser"): st.rerun()
        else:
            st.info("Base de données connectée. En attente des données du Scraper...")
