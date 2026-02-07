import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Vietnam Air Quality AI",
    page_icon="🇻🇳",
    layout="wide"
)

# --- 2. FONCTIONS DE CHARGEMENT (ROBUSTES) ---

@st.cache_data
def load_archive_data():
    """
    Charge le CSV historique en cherchant le fichier de manière intelligente.
    """
    # Liste des endroits probables
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "data/aqi_data.csv",                        # Chemin standard
        os.path.join(current_dir, "aqi_data.csv"),  # Même dossier
        "src/data/aqi_data.csv",                    # Dossier src
        "aqi_data.csv"                              # Racine
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
        # Nettoyage et typage
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # On force les colonnes numériques pour éviter les erreurs de type
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    else:
        return None

def init_connection():
    """Connexion à NeonDB via st.connection"""
    try:
        # Utilise les secrets configurés dans Streamlit Cloud
        return st.connection("postgresql", type="sql")
    except Exception as e:
        return None

def load_live_data(conn):
    """Récupère les 1000 dernières lignes de la base de données"""
    try:
        return conn.query('SELECT * FROM aqi_data ORDER BY timestamp DESC LIMIT 1000;', ttl="10m")
    except:
        return pd.DataFrame()

def get_aqi_color(aqi):
    """Retourne la couleur officielle de l'AQI"""
    if aqi <= 50: return "#00E400"  # Vert
    elif aqi <= 100: return "#FFFF00" # Jaune
    elif aqi <= 150: return "#FF7E00" # Orange
    elif aqi <= 200: return "#FF0000" # Rouge
    elif aqi <= 300: return "#8F3F97" # Violet
    else: return "#7E0023" # Marron

# --- 3. INTERFACE PRINCIPALE ---

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Aller vers :", ["📊 Archives & Performance IA", "🔴 Live Data (Temps Réel)"])

st.sidebar.markdown("---")
st.sidebar.info("Projet : Vietnam Air Quality Forecasting")

# ==============================================================================
# ONGLET 1 : ARCHIVES & IA (Le Labo)
# ==============================================================================
if page == "📊 Archives & Performance IA":
    st.title("🧠 Analyse de Performance du Modèle")
    st.markdown("""
    Cette section utilise les **données historiques** pour évaluer la précision de l'IA.
    Nous simulons des prédictions passées pour comparer la théorie (IA) et la réalité (Capteurs).
    """)

    df = load_archive_data()

    if df is not None:
        # --- A. Filtres ---
        col1, col2 = st.columns(2)
        with col1:
            locations = sorted(df['location'].unique())
            selected_location = st.selectbox("📍 Choisir une ville", locations)
        
        # Filtrer les données pour la ville choisie
        df_loc = df[df['location'] == selected_location].sort_values('timestamp')
        
        with col2:
            min_date = df_loc['timestamp'].min().date()
            max_date = df_loc['timestamp'].max().date()
            test_date = st.date_input("📅 Date à prédire (Test)", max_date, min_value=min_date, max_value=max_date)

        # --- B. Entraînement IA ---
        # On coupe : Tout ce qui est AVANT la date choisie sert à apprendre
        split_date = pd.Timestamp(test_date)
        train_df = df_loc[df_loc['timestamp'] < split_date]
        test_df = df_loc[df_loc['timestamp'].dt.date == split_date]

        if len(train_df) > 50 and len(test_df) > 0:
            with st.spinner('L\'IA analyse le passé et s\'entraîne...'):
                features = ['pm25', 'no2', 'so2', 'co', 'o3']
                # Vérification des colonnes disponibles
                features = [f for f in features if f in df.columns]
                
                X_train = train_df[features]
                y_train = train_df['aqi']
                X_test = test_df[features]
                y_test = test_df['aqi']

                # Création et entraînement du modèle Random Forest
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Prédiction
                y_pred = model.predict(X_test)

            # --- C. Résultats ---
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.markdown("### 🎯 Résultats de la simulation")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Date Analysée", str(test_date))
            kpi2.metric("Précision (R²)", f"{r2:.2f}", help="Plus proche de 1 est mieux")
            kpi3.metric("Erreur Moyenne (MAE)", f"{mae:.1f}", help="Écart moyen entre prédiction et réalité")

            # Graphique Interactif
            st.subheader(f"📉 Comparatif : Réalité vs Prédiction à {selected_location}")
            fig = go.Figure()
            
            # Courbe Réalité
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'], y=y_test,
                mode='lines+markers', name='Réalité (Mesuré)',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Courbe Prédiction
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'], y=y_pred,
                mode='lines', name='Prédiction IA',
                line=dict(color='#ff7f0e', width=3, dash='dot')
            ))
            
            fig.update_layout(xaxis_title="Heure", yaxis_title="AQI (Indice Qualité Air)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # Importance des variables
            with st.expander("Voir les polluants les plus influents"):
                importance = pd.DataFrame({
                    'Polluant': features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=True)
                fig_imp = px.bar(importance, x='Importance', y='Polluant', orientation='h')
                st.plotly_chart(fig_imp)

        else:
            st.warning("⚠️ Pas assez de données historiques AVANT cette date pour entraîner l'IA. Essayez une date plus récente.")
    
    else:
        st.error("❌ Le fichier CSV est introuvable malgré la recherche automatique.")

# ==============================================================================
# ONGLET 2 : LIVE DATA (NeonDB)
# ==============================================================================
elif page == "🔴 Live Data (Temps Réel)":
    st.title("📡 Monitoring Temps Réel (NeonDB)")
    st.markdown("Données en direct depuis la base de données Cloud.")
    
    conn = init_connection()
    
    if conn:
        df_live = load_live_data(conn)
        
        if not df_live.empty:
            # Traitement des dates
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
            latest_time = df_live['timestamp'].max()
            
            st.success(f"✅ Dernière donnée reçue : {latest_time}")
            
            # Affichage des 4 villes principales (KPIs)
            st.subheader("🌍 Situation Actuelle")
            latest_data = df_live.sort_values('timestamp', ascending=False).drop_duplicates('location').head(4)
            
            cols = st.columns(len(latest_data))
            for idx, row in enumerate(latest_data.itertuples()):
                with cols[idx]:
                    try:
                        val_aqi = int(row.aqi)
                        color = get_aqi_color(val_aqi)
                        st.markdown(f"""
                            <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; color: black;">
                                <h3 style="margin:0">{row.location}</h3>
                                <h1 style="font-size:3em; margin:0">{val_aqi}</h1>
                                <p style="margin:0">AQI</p>
                            </div>
                        """, unsafe_allow_html=True)
                    except:
                        st.error(f"Erreur données {row.location}")
            
            st.markdown("### 📋 Données Brutes Récentes")
            st.dataframe(df_live, use_container_width=True)
            
            if st.button("🔄 Actualiser les données"):
                st.rerun()
        else:
            st.info("La base de données est connectée mais ne contient pas encore de données récentes.")
            st.markdown("Le scraper automatique ajoutera les prochaines données à l'heure pile.")
    else:
        st.error("❌ Impossible de se connecter à la base de données. Vérifiez les 'Secrets' dans Streamlit.")
