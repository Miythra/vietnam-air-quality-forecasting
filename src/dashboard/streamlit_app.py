import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Vietnam Air Quality AI", page_icon="🇻🇳", layout="wide")

# --- 2. CHARGEMENT ROBUSTE (FIX DATE) ---
@st.cache_data
def load_archive_data():
    """
    Charge le CSV et convertit tout en HEURE VIETNAM (UTC+7).
    """
    import os
    
    # Recherche du fichier
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "data/aqi_data.csv", "src/data/aqi_data.csv",
        os.path.join(current_dir, "aqi_data.csv"), "aqi_data.csv"
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
        # --- CONVERSION TEMPORELLE VIETNAM ---
        # 1. On lit en UTC (Universel)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # 2. On convertit en heure locale du Vietnam (Asia/Ho_Chi_Minh)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh')
        
        # 3. On enlève l'info de fuseau pour le graphique (mais l'heure reste celle du Vietnam !)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        # Conversion chiffres
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

# ==============================================================================
# ONGLET 1 : ARCHIVES (CORRIGÉ)
# ==============================================================================
if page == "📊 Archives & Performance IA":
    st.title("🧠 Performance du Modèle")
    
    df = load_archive_data()

    if df is not None:
        # Sélecteurs
        col1, col2 = st.columns(2)
        with col1:
            locations = sorted(df['location'].unique())
            selected_location = st.selectbox("📍 Ville", locations)
            # Filtre les données pour cette ville
            df_loc = df[df['location'] == selected_location]
        
        with col2:
            # Récupère min et max
            min_date = df_loc['timestamp'].min().date()
            max_date = df_loc['timestamp'].max().date()
            
            # Sélecteur de date sécurisé
            try:
                test_date = st.date_input("📅 Date à prédire", max_date, min_value=min_date, max_value=max_date)
            except:
                st.warning("Plage de dates invalide, utilisation de la date max.")
                test_date = max_date

        # --- LOGIQUE INTELLIGENTE DE DÉCOUPE ---
        split_ts = pd.Timestamp(test_date)
        
        # 1. Essai Standard : On coupe à minuit (Passé vs Jour J)
        train_df = df_loc[df_loc['timestamp'] < split_ts]
        test_df = df_loc[df_loc['timestamp'].dt.date == test_date]
        
        mode_msg = "Mode Standard (Jours précédents vs Jour J)"

        # 2. PLAN B (Si pas assez de passé) : On coupe la journée en 2 (Matin vs Soir)
        # C'est ce qui va sauver ton affichage si tu n'as qu'un jour de données !
        if len(train_df) < 10:
            mode_msg = "⚠️ Mode 'Jour Unique' : Coupure 80% (Entraînement) / 20% (Test)"
            # On prend toutes les données du jour choisi
            day_data = df_loc[df_loc['timestamp'].dt.date == test_date].sort_values('timestamp')
            
            if len(day_data) > 5:
                split_idx = int(len(day_data) * 0.8) # 80% pour apprendre
                train_df = day_data.iloc[:split_idx]
                test_df = day_data.iloc[split_idx:]
            else:
                train_df = pd.DataFrame() # Vraiment pas assez de données

        # --- ENTRAÎNEMENT ---
        if len(train_df) > 5 and len(test_df) > 0:
            st.info(f"ℹ️ {mode_msg} | Entraînement sur {len(train_df)} lignes.")
            
            features = ['pm25', 'no2', 'so2', 'co', 'o3']
            features = [f for f in features if f in df.columns]
            
            X_train = train_df[features]
            y_train = train_df['aqi']
            X_test = test_df[features]
            y_test = test_df['aqi']

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- RÉSULTATS ---
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Date", str(test_date))
            kpi2.metric("Précision (R²)", f"{r2:.2f}")
            kpi3.metric("Erreur (MAE)", f"{mae:.1f}")

            # Graphique
            st.subheader("📉 Réalité vs Prédiction")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df['timestamp'], y=y_test, mode='lines+markers', name='Réalité'))
            fig.add_trace(go.Scatter(x=test_df['timestamp'], y=y_pred, mode='lines', name='Prédiction IA', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("❌ Données insuffisantes pour cette ville à cette date (même en mode secours).")
            st.write(f"Lignes trouvées : {len(df_loc)}")

    else:
        st.error("Fichier CSV introuvable.")

# ==============================================================================
# ONGLET 2 : LIVE DATA
# ==============================================================================
elif page == "🔴 Live Data (Temps Réel)":
    st.title("📡 Monitoring NeonDB")
    conn = init_connection()
    if conn:
        df_live = load_live_data(conn)
        if not df_live.empty:
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
            st.success(f"Dernière donnée : {df_live['timestamp'].max()}")
            st.dataframe(df_live.head(50), use_container_width=True)
            if st.button("Actualiser"): st.rerun()
        else:
            st.info("Base de données connectée mais vide.")
