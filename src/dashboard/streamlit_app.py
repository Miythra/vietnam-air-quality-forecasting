import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Vietnam Air Quality AI",
    page_icon="🇻🇳",
    layout="wide"
)

# --- 2. FONCTIONS UTILITAIRES ---

def get_aqi_color(aqi):
    """Retourne la couleur standard AQI"""
    if aqi <= 50: return "#00E400"  # Good (Green)
    elif aqi <= 100: return "#FFFF00" # Moderate (Yellow)
    elif aqi <= 150: return "#FF7E00" # Unhealthy for Sensitive (Orange)
    elif aqi <= 200: return "#FF0000" # Unhealthy (Red)
    elif aqi <= 300: return "#8F3F97" # Very Unhealthy (Purple)
    else: return "#7E0023" # Hazardous (Maroon)

@st.cache_data
def load_archive_data():
    """
    Charge le CSV historique.
    Cherche dans plusieurs dossiers possibles pour éviter les erreurs de chemin.
    """
    possible_paths = [
        "data/aqi_data.csv",           # Chemin standard demandé
        "src/data/aqi_data.csv",       # Autre structure commune
        "aqi_data.csv"                 # Racine
    ]
    
    df = None
    used_path = ""

    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                used_path = path
                break
            except:
                continue
    
    if df is not None:
        # Nettoyage et conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # On force les colonnes numériques
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    else:
        return None

def init_connection():
    """Connexion à NeonDB pour le Live"""
    return st.connection("postgresql", type="sql")

def load_live_data(conn):
    """Récupère les données fraiches de Neon"""
    try:
        # On ne charge que les 1000 dernières lignes pour aller vite
        return conn.query('SELECT * FROM aqi_data ORDER BY timestamp DESC LIMIT 1000;', ttl="10m")
    except:
        return pd.DataFrame() # Retourne vide si erreur ou table vide

# --- 3. INTERFACE & NAVIGATION ---

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Aller vers :", ["📊 Archives & Performance IA", "🔴 Live Data (Temps Réel)"])

st.sidebar.markdown("---")
st.sidebar.info("Projet : Vietnam Air Quality Forecasting")

# ==============================================================================
# PAGE 1 : ARCHIVES & PERFORMANCE (Le Laboratoire IA)
# ==============================================================================
if page == "📊 Archives & Performance IA":
    st.title("🧠 Analyse de Performance du Modèle")
    st.markdown("""
    Cette section utilise les **données historiques (CSV)** pour prouver l'efficacité de l'IA.
    Nous voyageons dans le passé pour voir si le modèle aurait pu prédire la pollution réelle.
    """)

    df = load_archive_data()

    if df is not None:
        # --- A. Filtres ---
        col_filters1, col_filters2 = st.columns(2)
        with col_filters1:
            locations = df['location'].unique()
            selected_location = st.selectbox("📍 Choisir une ville", locations)
        
        # Filtrer par lieu
        df_loc = df[df['location'] == selected_location].sort_values('timestamp')
        
        with col_filters2:
            # Sélecteur de date intelligent
            min_date = df_loc['timestamp'].min().date()
            max_date = df_loc['timestamp'].max().date()
            test_date = st.date_input("📅 Date à prédire (Test)", max_date, min_value=min_date, max_value=max_date)

        # --- B. Simulation IA ---
        # On coupe les données : Tout ce qui est AVANT la date sert à apprendre
        split_date = pd.Timestamp(test_date)
        train_df = df_loc[df_loc['timestamp'] < split_date]
        test_df = df_loc[df_loc['timestamp'].dt.date == split_date]

        if len(train_df) > 100 and len(test_df) > 0:
            
            with st.spinner('L\'IA analyse le passé et génère ses prédictions...'):
                features = ['pm25', 'no2', 'so2', 'co', 'o3'] 
                target = 'aqi'
                
                # Vérifier que les colonnes existent
                available_features = [f for f in features if f in df.columns]
                
                X_train = train_df[available_features]
                y_train = train_df[target]
                X_test = test_df[available_features]
                y_test = test_df[target]

                # Entraînement Random Forest
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Prédiction
                y_pred = model.predict(X_test)
                test_df = test_df.copy()
                test_df['predicted_aqi'] = y_pred

            # --- C. Résultats ---
            
            # 1. Métriques
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.markdown("### 🎯 Score de Précision")
            col1, col2, col3 = st.columns(3)
            col1.metric("Date Analysée", f"{test_date}")
            col2.metric("Précision (R²)", f"{r2:.2f}", delta_color="normal", help="Proche de 1 = Parfait")
            col3.metric("Erreur Moyenne (MAE)", f"{mae:.1f}", delta="-Good" if mae < 15 else "inverse", help="Plus c'est bas, mieux c'est")

            # 2. Graphique Principal
            st.subheader(f"📉 Réalité vs Prédiction à {selected_location}")
            
            fig = go.Figure()
            # Ligne Réelle
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'], y=test_df['aqi'],
                mode='lines+markers', name='Réalité (Mesuré)',
                line=dict(color='#1f77b4', width=3)
            ))
            # Ligne Prédite
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'], y=test_df['predicted_aqi'],
                mode='lines', name='Prédiction IA',
                line=dict(color='#ff7f0e', width=3, dash='dot')
            ))
            fig.update_layout(xaxis_title="Heure", yaxis_title="AQI", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # 3. Explication (Feature Importance)
            with st.expander("Voir comment l'IA a réfléchi (Importance des Polluants)"):
                importance = pd.DataFrame({
                    'Polluant': available_features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(importance, x='Importance', y='Polluant', orientation='h', 
                                 title="Poids des polluants dans la décision", color='Importance')
                st.plotly_chart(fig_imp, use_container_width=True)

        else:
            if len(train_df) <= 100:
                st.warning("⚠️ Pas assez de données historiques AVANT cette date pour entraîner l'IA. Choisissez une date plus récente.")
            else:
                st.warning("⚠️ Pas de données disponibles pour la date exacte sélectionnée.")

    else:
        st.error("❌ Impossible de trouver le fichier 'aqi_data.csv'. Vérifiez qu'il est bien dans le dossier 'data/' sur GitHub.")

# ==============================================================================
# PAGE 2 : LIVE DATA (NeonDB)
# ==============================================================================
elif page == "🔴 Live Data (Temps Réel)":
    st.title("📡 Monitoring en Temps Réel")
    st.markdown("Connexion directe à la base de données **NeonDB**. Les données apparaissent ici dès qu'elles sont collectées par le scraper.")

    conn = init_connection()
    df_live = load_live_data(conn)

    if not df_live.empty:
        # Conversion date
        df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
        
        # Dernier relevé
        latest_time = df_live['timestamp'].max()
        st.success(f"Dernière mise à jour reçue : {latest_time}")

        # KPI Cards pour les villes principales
        st.subheader("🌍 Situation Actuelle")
        
        # On prend les données les plus récentes par ville
        latest_data = df_live.sort_values('timestamp', ascending=False).drop_duplicates('location').head(4)
        
        cols = st.columns(len(latest_data))
        for index, (i, row) in enumerate(latest_data.iterrows()):
            with cols[index]:
                aqi_val = row['aqi'] if pd.notna(row['aqi']) else 0
                color = get_aqi_color(aqi_val)
                st.markdown(f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 10px; color: black; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="margin:0;">{row['location']}</h3>
                    <h1 style="font-size: 3em; margin:0;">{int(aqi_val)}</h1>
                    <p style="margin:0;">AQI</p>
                </div>
                """, unsafe_allow_html=True)

        # Tableau de données brutes
        st.markdown("### 📝 Historique Récent (Live)")
        st.dataframe(df_live, use_container_width=True)
        
        if st.button("🔄 Actualiser maintenant"):
            st.rerun()

    else:
        # Affichage élégant quand la base est vide
        st.info("👋 Bienvenue sur le Dashboard Live !")
        st.warning("⏳ La base de données est actuellement en attente de données.")
        
        st.markdown("""
        ### Statut du système :
        * **Base de données :** Connectée ✅
        * **Table aqi_data :** Détectée ✅
        * **Données :** En attente du premier passage du robot 🤖
        
        Le scraper automatique va bientôt remplir cette page. En attendant, vous pouvez consulter l'onglet **"Archives & Performance IA"** pour voir le modèle travailler sur l'historique.
        """)
