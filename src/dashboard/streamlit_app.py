import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import psycopg
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import numpy as np

# --- Configuration de la Page ---
st.set_page_config(layout="wide", page_title="Vietnam AQI Sentinel", page_icon="🇻🇳")

# --- CSS Personnalisé (Look Pro) ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .metric-card { background-color: #f9f9f9; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); text-align: center;}
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# --- Coordonnées (Pour la carte) ---
CITY_COORDS = {
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542}, 'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
    'Da Nang': {'lat': 16.0544, 'lon': 108.2022}, 'Hai Phong': {'lat': 20.8449, 'lon': 106.6881},
    'Can Tho': {'lat': 10.0452, 'lon': 105.7469}, 'Nha Trang': {'lat': 12.2388, 'lon': 109.1967},
    'Hue': {'lat': 16.4637, 'lon': 107.5909}, 'Vinh Phuc': {'lat': 21.3083, 'lon': 105.6046},
    'Bac Ninh': {'lat': 21.1861, 'lon': 106.0763}, 'Quang Ninh': {'lat': 20.9500, 'lon': 107.0833},
    'Nam Dinh': {'lat': 20.4200, 'lon': 106.1683}, 'Thai Nguyen': {'lat': 21.5942, 'lon': 105.8481}
}

# --- 1. Chargement & Caching ---
@st.cache_data(ttl=300)
def load_data():
    load_dotenv()
    db_url = os.environ.get('POSTGRES_URL')
    if not db_url:
        if "POSTGRES_URL" in st.secrets: db_url = st.secrets["POSTGRES_URL"]
        elif "general" in st.secrets: db_url = st.secrets["general"]["POSTGRES_URL"]
    
    if not db_url:
        st.error("🚨 Base de données introuvable (Secrets manquants).")
        st.stop()

    query = "SELECT * FROM aqi_data ORDER BY timestamp DESC;"
    try:
        with psycopg.connect(db_url) as conn:
            df = pd.read_sql(query, conn)
        
        # Preprocessing
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Bangkok')
        
        numeric_cols = ['aqi', 'pm25', 'pm10', 'co', 'so2', 'no2', 'o3']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ajout colonnes temporelles
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        return df
    except Exception as e:
        st.error(f"Erreur SQL : {e}")
        return pd.DataFrame()

# --- 2. Fonction de Prédiction (Simulation ou Modèle) ---
def run_forecasting(df, city):
    """
    Simule une prédiction basée sur les données récentes.
    À remplacer par `model.predict()` si tu uploads ton fichier .pkl
    """
    city_data = df[df['location'] == city].sort_values('timestamp')
    if city_data.empty: return None, None, None

    # 1. Calculer la moyenne des dernières 24h (Réalité)
    last_24h = city_data.tail(24)
    current_aqi = last_24h['aqi'].iloc[-1]
    avg_24h = last_24h['aqi'].mean()

    # 2. "Prédiction" pour les prochaines 24h 
    # (Ici on utilise une moyenne pondérée + tendance récente comme proxy)
    trend = current_aqi - avg_24h
    predicted_next_24h = avg_24h + (trend * 0.5) # Amortissement
    
    # Générer une courbe prévisionnelle
    future_dates = [city_data['timestamp'].max() + timedelta(hours=i) for i in range(1, 13)]
    future_values = [current_aqi + (trend * 0.1 * i) + np.random.normal(0, 5) for i in range(1, 13)]
    
    forecast_df = pd.DataFrame({'timestamp': future_dates, 'aqi': future_values, 'type': 'Prédiction'})
    history_df = last_24h[['timestamp', 'aqi']].copy()
    history_df['type'] = 'Historique'
    
    return current_aqi, predicted_next_24h, pd.concat([history_df, forecast_df])

# --- MAIN APP ---
def main():
    # --- DEBUT DU BLOC TEMPORAIRE D'IMPORTATION (CORRIGÉ) ---
    st.sidebar.header("🛠️ Admin Zone")
    uploaded_file = st.sidebar.file_uploader("Uploader l'historique (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        if st.sidebar.button("Lancer l'importation Cloud"):
            with st.spinner("Le serveur Streamlit lit et nettoie le fichier..."):
                import pandas as pd
                from sqlalchemy import create_engine, text
                
                # 1. Lecture du fichier
                df = pd.read_csv(uploaded_file)
                
                # NETTOYAGE AGRESSIF (La partie importante !)
                # On supprime l'ID s'il existe
                if 'id' in df.columns: df = df.drop(columns=['id'])
                
                # On convertit la date
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # On FORCE la conversion en nombres pour toutes les colonnes de pollution
                # errors='coerce' va transformer les erreurs (texte, tirets...) en NaN (vide)
                cols_to_fix = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
                for col in cols_to_fix:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # On remplace les NaN par None pour SQL
                df = df.replace({float('nan'): None})
                
                st.info(f"Fichier nettoyé : {len(df)} lignes. Envoi vers Neon en cours...")
                
                # 2. Connexion
                db_url = st.secrets["POSTGRES_URL"].replace("sslmode=require", "sslmode=require")
                
                try:
                    engine = create_engine(db_url.replace("postgresql://", "postgresql+psycopg://"))
                    
                    # Envoi par paquets de 1000
                    chunk_size = 1000
                    total_chunks = (len(df) // chunk_size) + 1
                    my_bar = st.progress(0)
                    
                    for i, chunk in enumerate(range(0, len(df), chunk_size)):
                        batch = df.iloc[chunk:chunk+chunk_size]
                        batch.to_sql('aqi_data', engine, if_exists='append', index=False, method='multi')
                        my_bar.progress((i + 1) / total_chunks)
                        
                    st.success("🎉 SUCCÈS TOTAL ! Tout l'historique est validé et importé.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Erreur d'importation : {e}")
    # --- FIN DU BLOC TEMPORAIRE ---
    # Load Data
    with st.spinner('Connexion au flux de données...'):
        df_raw = load_data()

    if df_raw.empty: return

    # --- SIDEBAR ---
    st.sidebar.title("🎛️ Contrôles")
    
    # Sélecteur de Ville (Global)
    cities = sorted(df_raw['location'].unique())
    if 'Hanoi' in cities: index_hanoi = cities.index('Hanoi')
    else: index_hanoi = 0
    selected_city = st.sidebar.selectbox("📍 Ville Cible", cities, index=index_hanoi)
    
    # Status Système
    last_update = df_raw['timestamp'].max()
    now = datetime.now(pytz.timezone('Asia/Bangkok'))
    is_live = (now - last_update) < timedelta(hours=4)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("État du Système")
    if is_live:
        st.sidebar.success(f"🟢 EN LIGNE\n\nMaj: {last_update.strftime('%H:%M')}")
    else:
        st.sidebar.error(f"🔴 OFF-LINE\n\nArrêt: {last_update.strftime('%d/%m %H:%M')}")

    # --- TABS STRUCTURE ---
    st.title(f"🇻🇳 Analyse Qualité de l'Air : {selected_city}")
    
    tab1, tab2, tab3 = st.tabs(["🔴 Surveillance Live", "📊 Analyse Approfondie", "🔮 Prévisions & IA"])

    # === TAB 1: SURVEILLANCE LIVE ===
    with tab1:
        # Filtrer pour la ville
        city_df = df_raw[df_raw['location'] == selected_city]
        latest = city_df.iloc[0] # Dernière ligne (triée par desc)

        # 1. Jauge AQI (Gauge Chart)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = latest['aqi'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AQI Actuel"},
                delta = {'reference': city_df.iloc[1]['aqi'], 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 300]},
                    'bar': {'color': "black", 'thickness': 0.05}, # Petite aiguille noire
                    'steps': [
                        {'range': [0, 50], 'color': "#00e400"},
                        {'range': [50, 100], 'color': "#ffff00"},
                        {'range': [100, 150], 'color': "#ff7e00"},
                        {'range': [150, 200], 'color': "#ff0000"},
                        {'range': [200, 300], 'color': "#8f3f97"}],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Afficher les polluants clés
            st.markdown("#### Polluants (µg/m³)")
            c1, c2 = st.columns(2)
            c1.metric("PM2.5", f"{latest['pm25']:.1f}")
            c2.metric("NO2", f"{latest['no2']:.1f}")

        with col2:
            # Carte centrée sur la ville
            st.subheader(f"🗺️ Localisation : {selected_city}")
            map_data = df_raw.groupby('location')[['aqi', 'pm25']].mean().reset_index()
            map_data['lat'] = map_data['location'].apply(lambda x: CITY_COORDS.get(x, {}).get('lat'))
            map_data['lon'] = map_data['location'].apply(lambda x: CITY_COORDS.get(x, {}).get('lon'))
            map_data = map_data.dropna()
            
            # On met en surbrillance la ville choisie
            fig_map = px.scatter_mapbox(
                map_data, lat="lat", lon="lon", color="aqi", size="aqi",
                color_continuous_scale="RdYlGn_r", size_max=40, zoom=6,
                center=CITY_COORDS.get(selected_city, {'lat':16, 'lon':106}),
                hover_name="location", mapbox_style="carto-positron"
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # === TAB 2: ANALYSE APPROFONDIE (Les graphes intéressants) ===
    with tab2:
        st.subheader("🔎 Corrélations et Profils Historiques")
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            # 1. Heatmap de Corrélation (Interactive)
            st.markdown("**Matrice de Corrélation (Tous polluants)**")
            corr = city_df[['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3']].corr()
            fig_corr = px.imshow(
                corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                title=f"Corrélations à {selected_city}"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with c_right:
            # 2. Profil de Pollution (Bar Chart)
            st.markdown("**Profil Moyen de Pollution**")
            avg_pol = city_df[['pm25', 'pm10', 'no2', 'so2', 'o3']].mean().reset_index()
            avg_pol.columns = ['Polluant', 'Concentration']
            fig_prof = px.bar(
                avg_pol, x='Concentration', y='Polluant', orientation='h',
                color='Concentration', color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_prof, use_container_width=True)
            
        # 3. Time Series avec Range Slider (Zoom)
        st.markdown("### 📅 Historique Complet (Zoomable)")
        fig_ts = px.line(city_df, x='timestamp', y=['aqi', 'pm25'], title="Evolution Temporelle")
        fig_ts.update_xaxes(rangeslider_visible=True) # LE SLIDER MAGIQUE
        st.plotly_chart(fig_ts, use_container_width=True)

    # === TAB 3: FORECASTING (La partie "Lancer le programme") ===
    with tab3:
        st.subheader("🔮 Prédiction de l'AQI")
        st.info("Ce module utilise les données historiques pour projeter la tendance des prochaines 12h.")
        
        col_act, col_pred = st.columns([1, 3])
        
        with col_act:
            if st.button("🚀 LANCER L'ANALYSE", type="primary"):
                with st.spinner("Calcul des tendances en cours..."):
                    current, pred, forecast_df = run_forecasting(df_raw, selected_city)
                
                # Metrics
                st.metric("AQI Actuel", f"{current:.0f}")
                
                delta = pred - current
                st.metric("Tendance (24h)", f"{pred:.0f}", delta=f"{delta:.1f}", delta_color="inverse")
                
                if pred > 150:
                    st.error("⚠️ Prévision : Qualité de l'air MAUVAISE. Port du masque recommandé.")
                elif pred > 100:
                    st.warning("⚠️ Prévision : Qualité MÉDIOCRE pour les groupes sensibles.")
                else:
                    st.success("✅ Prévision : Qualité de l'air ACCEPTABLE.")
            else:
                st.write("Cliquez pour générer la prévision.")

        with col_pred:
            if 'forecast_df' in locals() and forecast_df is not None:
                # Graphique Prédictif
                fig_pred = px.line(
                    forecast_df, x='timestamp', y='aqi', color='type',
                    color_discrete_map={'Historique': 'gray', 'Prédiction': 'blue'},
                    markers=True, title="Projection AQI (12 Prochaines heures)"
                )
                # Zone de confiance (Fake visuals for demo)
                fig_pred.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.05, line_width=0)
                fig_pred.add_hrect(y0=150, y1=300, fillcolor="red", opacity=0.05, line_width=0)
                
                st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main()
