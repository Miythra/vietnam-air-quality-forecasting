import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Vietnam Air Quality AI",
    page_icon="🇻🇳",
    layout="wide"
)

# --- FONCTIONS UTILITAIRES ---

def get_aqi_color(aqi):
    if aqi <= 50: return "#00E400"  # Good (Green)
    elif aqi <= 100: return "#FFFF00" # Moderate (Yellow)
    elif aqi <= 150: return "#FF7E00" # Unhealthy for Sensitive (Orange)
    elif aqi <= 200: return "#FF0000" # Unhealthy (Red)
    elif aqi <= 300: return "#8F3F97" # Very Unhealthy (Purple)
    else: return "#7E0023" # Hazardous (Maroon)

# --- CHARGEMENT DES DONNÉES ---

@st.cache_data
def load_archive_data():
    """Charge le CSV local pour la partie Analyse/Archive"""
    try:
        # Assure-toi que le fichier s'appelle bien 'aqi_data.csv' et est à la racine ou dans src
        df = pd.read_csv('aqi_data.csv') 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Nettoyage rapide
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        return None

def init_connection():
    """Connexion à NeonDB pour le Live"""
    return st.connection("postgresql", type="sql")

def load_live_data(conn):
    """Récupère les données fraiches de Neon"""
    try:
        return conn.query('SELECT * FROM aqi_data ORDER BY timestamp DESC;', ttl="10m")
    except:
        return pd.DataFrame() # Retourne vide si erreur (table vide)

# --- NAVIGATION ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Aller vers :", ["📊 Archives & Performance IA", "🔴 Live Data (Temps Réel)"])

# ==============================================================================
# PAGE 1 : ARCHIVES & PERFORMANCE (Le CSV)
# ==============================================================================
if page == "📊 Archives & Performance IA":
    st.title("🧠 Analyse de Performance du Modèle")
    st.markdown("""
    Cette section utilise les **données historiques** pour évaluer la capacité de l'IA à prédire la qualité de l'air.
    Nous simulons des prédictions passées pour comparer la **théorie vs la réalité**.
    """)

    df = load_archive_data()

    if df is not None:
        # 1. Filtres Latéraux
        st.sidebar.header("Paramètres de Simulation")
        locations = df['location'].unique()
        selected_location = st.sidebar.selectbox("📍 Choisir un lieu", locations)
        
        # Filtrer par lieu
        df_loc = df[df['location'] == selected_location].sort_values('timestamp')
        
        # Sélecteur de date pour le test
        min_date = df_loc['timestamp'].min().date()
        max_date = df_loc['timestamp'].max().date()
        
        st.sidebar.info(f"Données disponibles du {min_date} au {max_date}")
        test_date = st.sidebar.date_input("📅 Date à prédire", max_date, min_value=min_date, max_value=max_date)

        # 2. Préparation du Modèle (Entraînement à la volée pour la démo)
        # On entraîne sur tout ce qui est AVANT la date choisie
        split_date = pd.Timestamp(test_date)
        train_df = df_loc[df_loc['timestamp'] < split_date]
        test_df = df_loc[df_loc['timestamp'].dt.date == split_date]

        if len(train_df) > 50 and len(test_df) > 0:
            features = ['pm25', 'no2', 'so2', 'co', 'o3'] # On utilise les polluants pour prédire l'AQI
            target = 'aqi'
            
            X_train = train_df[features]
            y_train = train_df[target]
            X_test = test_df[features]
            y_test = test_df[target]

            # Entraînement rapide
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            test_df['predicted_aqi'] = y_pred

            # 3. Affichage des Résultats
            
            # Métriques Clés
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Date Analysée", f"{test_date}")
            col2.metric("Précision du Modèle (R²)", f"{r2:.2f}", help="1.0 est parfait, 0.0 est nul")
            col3.metric("Erreur Moyenne (MAE)", f"{mae:.1f}", help="Écart moyen entre prédiction et réalité")

            # Graphique : Réalité vs Prédiction
            st.subheader("📉 Comparatif : Réalité vs Prédiction IA")
            
            fig = go.Figure()
            
            # Ligne Réelle
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'], 
                y=test_df['aqi'],
                mode='lines+markers',
                name='Réalité (Mesuré)',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Ligne Prédite
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'], 
                y=test_df['predicted_aqi'],
                mode='lines',
                name='Prédiction IA',
                line=dict(color='#ff7f0e', width=3, dash='dot')
            ))
            
            fig.update_layout(title=f"Evolution de l'AQI à {selected_location} le {test_date}",
                              xaxis_title="Heure", yaxis_title="AQI", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Feature Importance (Sur quoi le modèle se base ?)
            st.subheader("🧪 Facteurs d'influence")
            st.markdown("Quels polluants ont le plus pesé dans la décision de l'IA ?")
            
            importance = pd.DataFrame({
                'Polluant': features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            fig_imp = px.bar(importance, x='Importance', y='Polluant', orientation='h', 
                             title="Importance des variables", color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.warning("⚠️ Pas assez de données pour cette date ou date située au tout début de l'historique. Choisissez une date plus récente.")

    else:
        st.error("❌ Fichier 'aqi_data.csv' introuvable. Veuillez l'uploader à la racine du projet.")

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
        latest = df_live.iloc[0]
        st.info(f"Dernière mise à jour : {latest['timestamp']}")

        # KPI Cards pour les villes principales
        st.subheader("🌍 Situation Actuelle")
        cols = st.columns(4)
        
        # On prend les 4 villes les plus récentes
        recent_cities = df_live.drop_duplicates(subset=['location']).head(4)
        
        for index, (i, row) in enumerate(recent_cities.iterrows()):
            with cols[index]:
                color = get_aqi_color(row['aqi'])
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 10px; color: black; text-align: center;">
                    <h3>{row['location']}</h3>
                    <h1>{int(row['aqi'])}</h1>
                    <p>AQI</p>
                </div>
                """, unsafe_allow_html=True)

        # Tableau de données brutes
        st.subheader("📝 Derniers relevés reçus")
        st.dataframe(df_live)
        
        # Bouton refresh manuel
        if st.button("🔄 Actualiser les données"):
            st.rerun()

    else:
        st.container()
        st.warning("⏳ La base de données est actuellement vide.")
        st.markdown("""
        ### Pourquoi est-ce vide ?
        C'est normal ! Vous venez de créer une nouvelle infrastructure.
        * Le **Scraper** va s'exécuter automatiquement à la prochaine heure programmée.
        * Dès que le premier relevé sera capturé, cette page s'animera automatiquement.
        
        Revenez dans une heure pour voir les premiers points apparaître ! 🌱
        """)
