import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Vietnam AQI - Maps & Predictions",
    page_icon="🇻🇳",
    layout="wide"
)

# --- COORDONNÉES DES VILLES (SECOURS) ---
# Si le CSV n'a pas de lat/lon, on utilise ça pour afficher la carte.
CITY_COORDS = {
    "Hanoi": [21.0285, 105.8542],
    "Ho Chi Minh": [10.8231, 106.6297],
    "Da Nang": [16.0544, 108.2022],
    "Hai Phong": [20.8449, 106.6881],
    "Can Tho": [10.0452, 105.7469],
    "Nha Trang": [12.2388, 109.1967],
    "Hue": [16.4637, 107.5909],
    "Ha Long": [20.9069, 107.0734],
    "Vung Tau": [10.3460, 107.0843],
    "Da Lat": [11.9404, 108.4583],
    "Bien Hoa": [10.9574, 106.8427],
    "Buon Ma Thuot": [12.6675, 108.0383]
}

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    possible_paths = ["data/aqi_data.csv", "src/data/aqi_data.csv", "aqi_data.csv"]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except: continue
    
    if df is not None:
        # Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        
        # Nombres
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['aqi'])
        
        # Ajout Lat/Lon si manquant
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            df['latitude'] = df['location'].map(lambda x: CITY_COORDS.get(x, [None, None])[0])
            df['longitude'] = df['location'].map(lambda x: CITY_COORDS.get(x, [None, None])[1])
            
        return df.dropna(subset=['latitude', 'longitude']).sort_values('timestamp')
    return None

# --- FONCTION DE PRÉDICTION GLOBALE ---
@st.cache_resource
def get_predictions_for_map(df, target_timestamp):
    """
    Entraîne un modèle pour chaque ville et prédit l'AQI à l'heure demandée.
    """
    map_data = []
    
    # On trouve l'heure la plus proche dans les données pour éviter d'être vide
    # (Tolérance de 1 heure)
    nearest_time = None
    min_diff = pd.Timedelta(hours=1)
    
    # On filtre d'abord grossièrement pour ne pas chercher dans tout l'historique
    subset_time = df[(df['timestamp'] >= target_timestamp - pd.Timedelta(hours=2)) & 
                     (df['timestamp'] <= target_timestamp + pd.Timedelta(hours=2))]
    
    if subset_time.empty:
        return pd.DataFrame(), None

    # Pour chaque ville présente
    for city in df['location'].unique():
        city_df = df[df['location'] == city].sort_values('timestamp')
        
        # On entraîne sur tout l'historique de cette ville
        features = ['pm25', 'no2', 'so2', 'co', 'o3']
        features = [f for f in features if f in city_df.columns]
        
        if len(city_df) > 5 and features:
            # Récupération de la ligne la plus proche de l'heure cible
            # On cherche la ligne réelle pour comparer
            row_closest = city_df.iloc[(city_df['timestamp'] - target_timestamp).abs().argmin()]
            time_diff = abs(row_closest['timestamp'] - target_timestamp)
            
            if time_diff < pd.Timedelta(hours=2): # On accepte max 2h d'écart
                real_aqi = row_closest['aqi']
                lat, lon = row_closest['latitude'], row_closest['longitude']
                
                # Entraînement Modèle (Rapide)
                # On exclut la ligne cible du train pour être honnête (ou pas, selon besoin)
                # Ici on entraîne sur tout sauf la cible pour simuler une prédiction
                train_data = city_df[city_df['timestamp'] != row_closest['timestamp']]
                if len(train_data) > 0:
                    model = RandomForestRegressor(n_estimators=20, random_state=42)
                    model.fit(train_data[features], train_data['aqi'])
                    
                    # Prédiction sur les features de l'heure cible
                    pred_aqi = model.predict(pd.DataFrame([row_closest[features]]))[0]
                    
                    map_data.append({
                        'location': city,
                        'lat': lat, 'lon': lon,
                        'Real AQI': real_aqi,
                        'Predicted AQI': pred_aqi,
                        'timestamp': row_closest['timestamp']
                    })
                    nearest_time = row_closest['timestamp']

    return pd.DataFrame(map_data), nearest_time

# --- SCALE COULEUR AQI ---
def get_aqi_color_scale():
    # Vert -> Jaune -> Orange -> Rouge -> Violet -> Marron
    return [
        (0, "#00e400"),    # Good
        (50/500, "#ffff00"),# Moderate
        (100/500, "#ff7e00"),# Unhealthy for Sensitive
        (150/500, "#ff0000"),# Unhealthy
        (200/500, "#8f3f97"),# Very Unhealthy
        (300/500, "#7e0023"),# Hazardous
        (1, "#7e0023")
    ]

# --- INTERFACE ---
st.title("🇻🇳 Vietnam Air Quality: Reality vs AI Prediction")
st.markdown("Comparaison spatio-temporelle de la pollution de l'air.")

df = load_data()

if df is not None:
    # --- 1. SÉLECTEUR DE TEMPS (POUR LA CARTE) ---
    st.sidebar.header("🗺️ Carte Interactive")
    
    # Slider pour choisir l'heure exacte à afficher sur la carte
    min_date = df['timestamp'].min().to_pydatetime()
    max_date = df['timestamp'].max().to_pydatetime()
    
    selected_time = st.sidebar.slider(
        "Choisir le moment à visualiser :",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="DD/MM/YY - HH:mm"
    )

    # Calcul des données pour la carte
    map_df, exact_time = get_predictions_for_map(df, pd.Timestamp(selected_time))

    if not map_df.empty:
        st.info(f"Visualisation des données pour le : **{exact_time}**")
        
        col_map1, col_map2 = st.columns(2)
        
        # MAP GAUCHE : RÉALITÉ
        with col_map1:
            st.subheader("🌍 Réalité (Mesurée)")
            fig_real = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", size="Real AQI", color="Real AQI",
                color_continuous_scale=px.colors.cyclical.IceFire, # Ou custom AQI scale
                range_color=[0, 300],
                hover_name="location", hover_data={"Real AQI": True, "lat": False, "lon": False},
                zoom=4.5, center={"lat": 16.0, "lon": 106.0},
                mapbox_style="carto-positron",
                title="Vraies mesures"
            )
            st.plotly_chart(fig_real, use_container_width=True)

        # MAP DROITE : PRÉDICTION
        with col_map2:
            st.subheader("🤖 Prédiction IA")
            fig_pred = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", size="Predicted AQI", color="Predicted AQI",
                color_continuous_scale=px.colors.cyclical.IceFire,
                range_color=[0, 300],
                hover_name="location", hover_data={"Predicted AQI": True, "lat": False, "lon": False},
                zoom=4.5, center={"lat": 16.0, "lon": 106.0},
                mapbox_style="carto-positron",
                title="Estimations du Modèle"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning(f"Pas de données trouvées autour de {selected_time}. Essayez de bouger le curseur.")

    st.divider()

    # --- 2. GRAPHIQUE TEMPOREL AVANCÉ ---
    st.header("📈 Analyse Temporelle Détaillée")
    
    col_filter1, col_filter2 = st.columns(2)
    
    # Sélecteur Ville
    cities = sorted(df['location'].unique())
    selected_city = col_filter1.selectbox("📍 Choisir une ville pour l'analyse :", cities)
    
    # Sélecteur Plage de Dates
    date_range = col_filter2.date_input(
        "📅 Plage de dates :",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_d, end_d = date_range
        # Filtrage
        mask = (df['location'] == selected_city) & \
               (df['timestamp'].dt.date >= start_d) & \
               (df['timestamp'].dt.date <= end_d)
        df_chart = df[mask]
        
        if not df_chart.empty and len(df_chart) > 5:
            # Entraînement spécifique pour le graphe (pour avoir la courbe de prédiction complète)
            features = ['pm25', 'no2', 'so2', 'co', 'o3']
            features = [f for f in features if f in df_chart.columns]
            
            X = df_chart[features]
            y = df_chart['aqi']
            
            # On entraîne sur tout le set visible pour montrer le "fitting" (ajustement)
            # Ou on fait un split si tu veux montrer la prédiction pure.
            # Ici, pour visualiser "Réalité vs Prédiction", on montre souvent comment le modèle s'ajuste.
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            df_chart['Predicted'] = model.predict(X)
            
            # Graphique
            fig_line = go.Figure()
            
            # Zone AQI Background (Gradient visuel)
            # Astuce : On peut colorer le background, mais c'est chargé.
            # On va plutôt colorer la ligne ou utiliser des marqueurs.
            
            fig_line.add_trace(go.Scatter(
                x=df_chart['timestamp'], y=df_chart['aqi'],
                mode='lines+markers', name='Réalité',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig_line.add_trace(go.Scatter(
                x=df_chart['timestamp'], y=df_chart['Predicted'],
                mode='lines', name='Prédiction IA',
                line=dict(color='#ff7f0e', width=2, dash='dot')
            ))
            
            fig_line.update_layout(
                title=f"Evolution AQI à {selected_city}",
                xaxis_title="Date",
                yaxis_title="AQI",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Métriques sur la période sélectionnée
            mae = mean_absolute_error(df_chart['aqi'], df_chart['Predicted'])
            r2 = r2_score(df_chart['aqi'], df_chart['Predicted'])
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("AQI Max (Période)", int(df_chart['aqi'].max()))
            kpi2.metric("Précision Modèle (R²)", f"{r2:.2f}")
            kpi3.metric("Erreur Moyenne", f"{mae:.1f}")
            
        else:
            st.warning("Pas assez de données sur cette période pour afficher le graphique.")
    else:
        st.info("Veuillez sélectionner une date de début et de fin.")

else:
    st.error("Aucune donnée disponible. Veuillez vérifier le fichier CSV.")
