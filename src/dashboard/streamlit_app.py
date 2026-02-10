import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Vietnam AQI - AI Dashboard",
    page_icon="🇻🇳",
    layout="wide"
)

# --- STYLE CSS (Onglets & Cartes) ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- COORDONNÉES VILLES (Référence) ---
CITY_COORDS = {
    "Hanoi": [21.0285, 105.8542], "Ho Chi Minh": [10.8231, 106.6297],
    "Da Nang": [16.0544, 108.2022], "Hai Phong": [20.8449, 106.6881],
    "Can Tho": [10.0452, 105.7469], "Nha Trang": [12.2388, 109.1967],
    "Hue": [16.4637, 107.5909], "Ha Long": [20.9069, 107.0734],
    "Vung Tau": [10.3460, 107.0843], "Da Lat": [11.9404, 108.4583],
    "Bien Hoa": [10.9574, 106.8427], "Buon Ma Thuot": [12.6675, 108.0383],
    "Bac Giang": [21.2731, 106.1946], "Bac Ninh": [21.1861, 106.0763]
}

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    paths = ["data/aqi_data.csv", "src/data/aqi_data.csv", "aqi_data.csv"]
    df = None
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except: continue
    
    if df is not None:
        # Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        
        # Numérique
        for col in ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=['aqi']).sort_values('timestamp')
        
        # Injection Lat/Lon si manquant
        if 'latitude' not in df.columns:
            df['latitude'] = df['location'].map(lambda x: CITY_COORDS.get(x, [None, None])[0])
            df['longitude'] = df['location'].map(lambda x: CITY_COORDS.get(x, [None, None])[1])
            
        return df.dropna(subset=['latitude'])
    return None

# --- MOTEUR IA POUR LA CARTE ---
@st.cache_resource
def get_map_snapshot(df, target_time):
    """Génère les données Réelles vs Prédites pour TOUTES les villes à un instant T."""
    snapshot = []
    
    # On cherche les données +/- 2h autour de l'heure cible
    time_window = df[(df['timestamp'] >= target_time - pd.Timedelta(hours=2)) & 
                     (df['timestamp'] <= target_time + pd.Timedelta(hours=2))]
    
    if time_window.empty: return pd.DataFrame()

    for city in df['location'].unique():
        city_df = df[df['location'] == city]
        
        # Trouver la ligne la plus proche de l'heure cible
        if city_df.empty: continue
        
        # Index de la ligne la plus proche
        idx_closest = (city_df['timestamp'] - target_time).abs().idxmin()
        row = city_df.loc[idx_closest]
        
        # Si trop loin dans le temps (>3h), on ignore
        if abs(row['timestamp'] - target_time) > pd.Timedelta(hours=3):
            continue

        # Entraînement Modèle Rapide (Sur tout l'historique de la ville sauf ce point)
        features = ['pm25', 'no2', 'so2', 'co', 'o3']
        valid_feats = [f for f in features if f in city_df.columns and pd.notnull(row[f])]
        
        pred_aqi = None
        if len(city_df) > 10 and valid_feats:
            train_set = city_df[city_df.index != idx_closest].dropna(subset=valid_feats+['aqi'])
            if len(train_set) > 5:
                model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
                model.fit(train_set[valid_feats], train_set['aqi'])
                pred_aqi = model.predict(pd.DataFrame([row[valid_feats]]))[0]

        snapshot.append({
            'location': city,
            'lat': row['latitude'], 'lon': row['longitude'],
            'Real AQI': row['aqi'],
            'Predicted AQI': pred_aqi if pred_aqi else row['aqi'], # Fallback si pas de prédiction
            'timestamp': row['timestamp']
        })
    
    return pd.DataFrame(snapshot)

# --- COULEURS AQI ---
def get_color_scale():
    # Echelle officielle AQI (0-500)
    return [
        (0.0, "#00E400"),  # 0-50 Good
        (0.1, "#FFFF00"),  # 51-100 Moderate
        (0.2, "#FF7E00"),  # 101-150 Sensitive
        (0.3, "#FF0000"),  # 151-200 Unhealthy
        (0.4, "#8F3F97"),  # 201-300 Very Unhealthy
        (0.6, "#7E0023"),  # 300+ Hazardous
        (1.0, "#7E0023")
    ]

# === APP ===
st.title("🇻🇳 Vietnam Air Quality AI Center")
st.markdown("Plateforme unifiée : Cartographie Temps Réel (Simulé) & Analyse IA")

df = load_data()

if df is not None:
    # Onglets Principaux
    tab_map, tab_graph, tab_deep = st.tabs(["🗺️ Cartographie (Heatmap)", "📈 Analyse Temporelle", "🧠 Performance IA"])

    # =================================================
    # TAB 1 : CARTES (HEATMAP STYLE)
    # =================================================
    with tab_map:
        st.subheader("Visualisation Spatiale : Réalité vs IA")
        
        # Slider Temporel
        min_ts, max_ts = df['timestamp'].min().to_pydatetime(), df['timestamp'].max().to_pydatetime()
        selected_ts = st.slider("📅 Choisir le moment :", min_value=min_ts, max_value=max_ts, value=max_ts, format="DD/MM HH:mm")
        
        # Génération Données Carte
        map_df = get_map_snapshot(df, pd.Timestamp(selected_ts))
        
        if not map_df.empty:
            col_real, col_pred = st.columns(2)
            
            # Paramètres communs pour le look "Heatmap"
            map_config = dict(
                lat="lat", lon="lon",
                color_continuous_scale=get_color_scale(),
                range_color=[0, 300],
                zoom=4.8, center={"lat": 16.5, "lon": 106.5},
                mapbox_style="carto-positron",
                size_max=30
            )

            with col_real:
                st.markdown("**🌍 Mesures Réelles**")
                fig1 = px.scatter_mapbox(
                    map_df, color="Real AQI", size="Real AQI", # Taille varie avec pollution
                    hover_name="location", **map_config
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_pred:
                st.markdown("**🤖 Prédictions Modèle**")
                fig2 = px.scatter_mapbox(
                    map_df, color="Predicted AQI", size="Predicted AQI",
                    hover_name="location", **map_config
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            st.caption(f"Données affichées pour la date la plus proche de : {selected_ts}")
        else:
            st.warning("Pas assez de données disponibles autour de cette date.")

    # =================================================
    # TAB 2 : ANALYSE TEMPORELLE
    # =================================================
    with tab_graph:
        col_city, col_range = st.columns([1, 2])
        target_city = col_city.selectbox("Choisir une ville :", sorted(df['location'].unique()))
        
        # Filtre Données
        city_data = df[df['location'] == target_city]
        
        # Graphe
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['aqi'], mode='lines', name='AQI Réel', line=dict(color='#1f77b4', width=2)))
        fig_time.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['pm25'], mode='lines', name='PM2.5', line=dict(color='#ff7f0e', dash='dot')))
        
        fig_time.update_layout(title=f"Historique à {target_city}", hovermode="x unified", xaxis_title="Date", yaxis_title="Niveau")
        st.plotly_chart(fig_time, use_container_width=True)

    # =================================================
    # TAB 3 : DEEP DIVE IA (Scatter & Metrics)
    # =================================================
    with tab_deep:
        st.subheader(f"Diagnostique du Modèle pour {target_city}")
        
        # Entraînement à la volée pour l'analyse
        features = ['pm25', 'no2', 'so2', 'co', 'o3']
        feats = [f for f in features if f in city_data.columns]
        
        if len(city_data) > 10:
            X = city_data[feats]
            y = city_data['aqi']
            
            # Split 80/20 chrono
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # KPIs
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            k1, k2 = st.columns(2)
            k1.metric("Précision (R²)", f"{r2:.2f}")
            k2.metric("Erreur Moyenne (MAE)", f"{mae:.1f}")
            
            # Scatter Plot (Pred vs Real)
            fig_scat = px.scatter(x=y_test, y=y_pred, labels={'x': 'Réalité', 'y': 'Prédiction'}, title="Justesse des prédictions")
            fig_scat.add_shape(type="line", line=dict(dash='dash', color='grey'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            st.plotly_chart(fig_scat, use_container_width=True)
            
            # Feature Importance
            imp = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_}).sort_values('Importance')
            fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', title="Poids des polluants dans le calcul")
            st.plotly_chart(fig_imp, use_container_width=True)
            
        else:
            st.info("Pas assez de données pour l'analyse IA.")

else:
    st.error("Fichier CSV introuvable.")
