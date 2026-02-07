import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import psycopg
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

# --- Configuration de la Page ---
st.set_page_config(layout="wide", page_title="Vietnam AQI Watch", page_icon="🇻🇳")

# --- Coordonnées des villes (Ajout manuel pour la carte) ---
CITY_COORDS = {
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542},
    'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
    'Da Nang': {'lat': 16.0544, 'lon': 108.2022},
    'Hai Phong': {'lat': 20.8449, 'lon': 106.6881},
    'Can Tho': {'lat': 10.0452, 'lon': 105.7469},
    'Nha Trang': {'lat': 12.2388, 'lon': 109.1967},
    'Hue': {'lat': 16.4637, 'lon': 107.5909},
    'Vinh Phuc': {'lat': 21.3083, 'lon': 105.6046},
    'Bac Ninh': {'lat': 21.1861, 'lon': 106.0763},
    'Quang Ninh': {'lat': 20.9500, 'lon': 107.0833},
    # Ajoute d'autres villes ici si nécessaire, sinon elles n'apparaitront pas sur la carte
}

# --- 1. Chargement des Données (Blindé) ---
@st.cache_data(ttl=300) # Cache de 5 min pour ne pas surcharger la base
def load_data():
    load_dotenv()
    db_url = os.environ.get('POSTGRES_URL')
    
    # Gestion des secrets Streamlit Cloud
    if not db_url:
        if "POSTGRES_URL" in st.secrets:
            db_url = st.secrets["POSTGRES_URL"]
        elif "general" in st.secrets:
            db_url = st.secrets["general"]["POSTGRES_URL"]

    if not db_url:
        st.error("🚨 Clé Database introuvable.")
        st.stop()

    query = "SELECT * FROM aqi_data ORDER BY timestamp DESC;"
    
    try:
        with psycopg.connect(db_url) as conn:
            df = pd.read_sql(query, conn)
        
        # Nettoyage et Conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Conversion UTC vers Heure Vietnam (GMT+7)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Bangkok') if df['timestamp'].dt.tz else df['timestamp'] + pd.Timedelta(hours=7)
        
        # Numeric conversion
        numeric_cols = ['aqi', 'pm25', 'pm10', 'co', 'so2', 'no2', 'o3']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Erreur SQL: {e}")
        return pd.DataFrame()

# --- 2. Fonction Status Système (Le Countdown) ---
def display_system_status(df):
    if df.empty:
        return

    last_update = df['timestamp'].max()
    now = datetime.now(pytz.timezone('Asia/Bangkok'))
    diff = now - last_update
    
    # Style CSS pour les badges
    st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
    .live-badge { background-color: #d4edda; color: #155724; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .offline-badge { background-color: #f8d7da; color: #721c24; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if diff < timedelta(hours=2):
            # C'est LIVE !
            next_update = last_update + timedelta(hours=1)
            time_to_next = next_update - now
            minutes_left = int(time_to_next.total_seconds() / 60)
            if minutes_left < 0: minutes_left = 0
            
            st.markdown(f"""
            <div class="metric-card">
                <span class="live-badge">🟢 LIVE SYSTEM</span><br>
                <small>Dernière donnée : {last_update.strftime('%H:%M')}</small><br>
                <strong>Prochaine collecte dans ~{minutes_left} min</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            # C'est OFFLINE / Historique
            st.markdown(f"""
            <div class="metric-card">
                <span class="offline-badge">🔴 ARCHIVE MODE</span><br>
                <small>Le système de collecte est en pause.</small><br>
                Dernière mise à jour : <strong>{last_update.strftime('%d %b %Y à %H:%M')}</strong>
            </div>
            """, unsafe_allow_html=True)

# --- 3. Interface Principale ---
def main():
    # Chargement
    with st.spinner('Connexion à NeonDB...'):
        df_raw = load_data()

    if df_raw.empty:
        st.warning("Aucune donnée disponible.")
        return

    # Sidebar : Filtres Interactifs
    st.sidebar.header("🔍 Filtres & Contrôles")
    
    # A. Sélecteur de Date (Time Slider)
    min_date = df_raw['timestamp'].min().date()
    max_date = df_raw['timestamp'].max().date()
    
    selected_dates = st.sidebar.date_input(
        "Période d'analyse",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtrage du DataFrame
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_d, end_d = selected_dates
        mask = (df_raw['timestamp'].dt.date >= start_d) & (df_raw['timestamp'].dt.date <= end_d)
        df = df_raw.loc[mask]
    else:
        df = df_raw # Pas de filtre si erreur de sélection

    # B. Sélecteur de Ville
    locations = ['Toutes'] + sorted(df['location'].unique().tolist())
    selected_location = st.sidebar.selectbox("Choisir une ville", locations)

    # --- HEADER & STATUS ---
    st.title("🏭 Vietnam Air Quality Monitoring")
    display_system_status(df_raw) # On affiche le status basé sur les données brutes les plus récentes
    
    st.markdown("---")

    # --- VUE CARTE (Interactive Map) ---
    if selected_location == 'Toutes':
        st.subheader("🗺️ Vue Géographique (Moyenne sur la période)")
        
        # Préparer les données pour la carte
        map_df = df.groupby('location')[['aqi', 'pm25']].mean().reset_index()
        
        # Ajouter lat/lon
        map_df['lat'] = map_df['location'].apply(lambda x: CITY_COORDS.get(x, {}).get('lat'))
        map_df['lon'] = map_df['location'].apply(lambda x: CITY_COORDS.get(x, {}).get('lon'))
        map_df = map_df.dropna(subset=['lat', 'lon']) # Enlever villes sans coords
        
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df, 
                lat="lat", lon="lon", 
                color="aqi", size="aqi",
                hover_name="location",
                color_continuous_scale=px.colors.cyclical.IceFire,
                size_max=30, zoom=5,
                mapbox_style="carto-positron",
                title="Carte de Pollution Moyenne"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Pas assez de coordonnées GPS configurées pour afficher la carte.")

    # --- STATS & KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    current_df = df if selected_location == 'Toutes' else df[df['location'] == selected_location]
    
    avg_aqi = current_df['aqi'].mean()
    max_aqi = current_df['aqi'].max()
    dominant_pollutant = "PM2.5" # Simplification basée sur ton rapport
    
    col1.metric("AQI Moyen", f"{avg_aqi:.0f}", delta_color="inverse")
    col2.metric("Pic de Pollution", f"{max_aqi:.0f}")
    col3.metric("Polluant Principal", dominant_pollutant)
    col4.metric("Points de données", f"{len(current_df):,}")

    # --- GRAPHIQUES INTERACTIFS (Plotly) ---
    
    # 1. Time Series (Zoomable)
    st.subheader(f"📈 Évolution Temporelle : {selected_location}")
    
    if selected_location == 'Toutes':
        # Top 5 villes les plus polluées pour ne pas surcharger le graphe
        top_cities = df.groupby('location')['aqi'].mean().nlargest(5).index
        plot_df = df[df['location'].isin(top_cities)]
        title = "Top 5 Villes les plus polluées (Comparaison)"
    else:
        plot_df = current_df
        title = f"Historique AQI : {selected_location}"

    # Graphique linéaire interactif
    fig_line = px.line(
        plot_df, 
        x='timestamp', 
        y='aqi', 
        color='location',
        title=title,
        labels={'aqi': 'Indice AQI', 'timestamp': 'Date'},
        template="plotly_white"
    )
    # Ajouter des zones de couleur (Bon, Modéré, Mauvais)
    fig_line.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Bon")
    fig_line.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Modéré")
    fig_line.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Sensible")
    fig_line.add_hrect(y0=150, y1=300, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Mauvais")
    
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. Comparaison des Polluants (Bar Chart)
    st.subheader("🧪 Composition de la Pollution")
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    
    # Calcul des moyennes
    pol_means = current_df[pollutants].mean().reset_index()
    pol_means.columns = ['Polluant', 'Concentration']
    
    fig_bar = px.bar(
        pol_means, 
        x='Polluant', 
        y='Concentration', 
        color='Concentration',
        color_continuous_scale='Reds',
        title=f"Concentration Moyenne des Polluants ({selected_location})"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Données collectées via aqi.in • Projet Data Science Group 2")

if __name__ == "__main__":
    main()
