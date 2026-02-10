import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Vietnam Air Quality AI - Analytics",
    page_icon="📊",
    layout="wide"
)

# --- COORDONNÉES GPS COMPLÈTES (Basées sur ton CSV) ---
CITY_COORDS = {
    "Hanoi": [21.0285, 105.8542],
    "Ha Noi": [21.0285, 105.8542],
    "Ho Chi Minh": [10.8231, 106.6297],
    "Da Nang": [16.0544, 108.2022],
    "Hai Phong": [20.8449, 106.6881],
    "Can Tho": [10.0452, 105.7469],
    "Ba Ria Vung Tau": [10.4114, 107.1362],
    "Bac Giang": [21.2731, 106.1946],
    "Bac Ninh": [21.1861, 106.0763],
    "Binh Dinh": [13.7820, 109.2192],
    "Binh Duong": [10.9805, 106.6519],
    "Cao Bang": [22.6667, 106.2500],
    "Gia Lai": [13.9833, 108.0000],
    "Ha Nam": [20.5453, 105.9122],
    "Ha Nam Province": [20.5453, 105.9122],
    "Hai Duong": [20.9333, 106.3167],
    "Hung Yen": [20.6464, 106.0511],
    "Hoa Binh Province": [20.8133, 105.3383],
    "Khanh Hoa": [12.2388, 109.1967],
    "Lam Dong": [11.9404, 108.4583],
    "Lang Son": [21.8533, 106.7583],
    "Lang Son Province": [21.8533, 106.7583],
    "Lao Cai": [22.4833, 103.9667],
    "Long An": [10.5333, 106.4167],
    "Nghe An": [18.6667, 105.6667],
    "Ninh Binh": [20.2539, 105.9750],
    "Ninh Thuan": [11.5667, 108.9833],
    "Phu Tho": [21.3000, 105.4000],
    "Quang Binh Province": [17.4833, 106.6000],
    "Quang Nam": [15.5667, 108.4833],
    "Quang Ngai": [15.1167, 108.8000],
    "Quang Ninh": [20.9500, 107.0833],
    "Quang Ninh Province": [20.9500, 107.0833],
    "Quang Tri": [16.8000, 107.1000],
    "Son La": [21.3167, 103.9000],
    "Tay Ninh": [11.3000, 106.1167],
    "Thai Binh": [20.4500, 106.3333],
    "Thai Binh Province": [20.4500, 106.3333],
    "Thua Thien Hue": [16.4637, 107.5909],
    "Tra Vinh": [9.9333, 106.3333],
    "Tuyen Quang Province": [21.8167, 105.2167],
    "Vinh Long": [10.2500, 105.9667],
    "Vinh Phuc": [21.3000, 105.6000]
}

# --- STYLE CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

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
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna().sort_values('timestamp')
    return None

def get_aqi_color(aqi):
    if aqi <= 50: return "#00E400"
    elif aqi <= 100: return "#FFFF00"
    elif aqi <= 150: return "#FF7E00"
    elif aqi <= 200: return "#FF0000"
    elif aqi <= 300: return "#8F3F97"
    else: return "#7E0023"

# --- ENTRAÎNEMENT IA ---
@st.cache_resource
def train_model(df_loc):
    features = ['pm25', 'no2', 'so2', 'co', 'o3']
    features = [f for f in features if f in df_loc.columns]
    
    if len(df_loc) < 20 or not features:
        return None, None, None, None, None

    X = df_loc[features]
    y = df_loc['aqi']
    
    split_idx = int(len(df_loc) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df_loc['timestamp'].iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, dates_test

# --- MAIN APP ---
st.title("📊 Analyse Avancée & Performance IA")

df = load_data()

if df is not None:
    st.sidebar.header("🎛️ Configuration")
    locations = sorted(df['location'].unique())
    selected_location = st.sidebar.selectbox("📍 Ville cible", locations)
    
    # --- BLOC MAP ---
    st.subheader("🗺️ Situation Géographique (Moyennes du dernier jour)")
    
    latest_date = df['timestamp'].max().date()
    df_recent = df[df['timestamp'].dt.date == latest_date]
    
    if not df_recent.empty:
        daily_stats = df_recent.groupby('location')['aqi'].mean().reset_index()
        
        row_target = daily_stats[daily_stats['location'] == selected_location]
        row_best = daily_stats.loc[daily_stats['aqi'].idxmin()]
        row_worst = daily_stats.loc[daily_stats['aqi'].idxmax()]
        
        map_points = []
        
        def add_point(row, label_type):
            lat, lon = CITY_COORDS.get(row['location'], [None, None])
            if lat:
                map_points.append({
                    'location': row['location'],
                    'aqi': int(row['aqi']),
                    'lat': lat, 'lon': lon,
                    'type': label_type,
                    'size': 15 if label_type == 'Target' else 12,
                    'color': get_aqi_color(row['aqi'])
                })

        if not row_target.empty:
            add_point(row_target.iloc[0], '🎯 Cible (Target)')
        
        if row_best['location'] != selected_location:
            add_point(row_best, '✅ Meilleure (Best)')
        else:
             for p in map_points:
                 if p['location'] == selected_location: p['type'] += " & ✅ Best"

        if row_worst['location'] != selected_location:
            add_point(row_worst, '❌ Pire (Worst)')
        else:
             for p in map_points:
                 if p['location'] == selected_location: p['type'] += " & ❌ Worst"
        
        df_map = pd.DataFrame(map_points)
        
        if not df_map.empty:
            fig_map = px.scatter_mapbox(
                df_map, lat="lat", lon="lon", color="type", size="size",
                hover_name="location", hover_data={"aqi": True, "lat": False, "lon": False},
                zoom=5, center={"lat": 16.0, "lon": 106.0},
                mapbox_style="carto-positron", title=f"Aperçu du {latest_date}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            m1, m2, m3 = st.columns(3)
            if not row_target.empty:
                m1.metric(f"🎯 {selected_location}", f"{int(row_target.iloc[0]['aqi'])} AQI", "Votre sélection")
            m2.metric(f"✅ {row_best['location']}", f"{int(row_best['aqi'])} AQI", "Meilleur air")
            m3.metric(f"❌ {row_worst['location']}", f"{int(row_worst['aqi'])} AQI", "Pire air", delta_color="inverse")
        else:
            st.warning("⚠️ Carte vide : Les noms des villes dans le CSV ne correspondent pas aux coordonnées GPS.")

    st.divider()

    # --- DASHBOARD ---
    df_loc = df[df['location'] == selected_location]
    with st.spinner(f"Entraînement du modèle pour {selected_location}..."):
        model, X_test, y_test, y_pred, dates_test = train_model(df_loc)

    if model is not None:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Échantillons Testés", len(y_test), border=True)
        col2.metric("Précision (R²)", f"{r2:.2%}", delta_color="normal" if r2 > 0.7 else "inverse", border=True)
        col3.metric("Erreur Moyenne (MAE)", f"{mae:.1f}", delta="-Low is good", delta_color="inverse", border=True)
        col4.metric("RMSE (Erreur Quadratique)", f"{rmse:.1f}", border=True)

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Analyse Temporelle", "🎯 Précision & Corrélation", "📉 Analyse des Erreurs", "🧠 Intégration Modèle"])

        with tab1:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Réalité', line=dict(color='#1f77b4', width=2)))
            fig_ts.add_trace(go.Scatter(x=dates_test, y=y_pred, mode='lines', name='Prédiction IA', line=dict(color='#ff7f0e', width=2, dash='dot')))
            fig_ts.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="AQI", legend=dict(orientation="h", y=1.1), height=500)
            fig_ts.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_ts, use_container_width=True)

        with tab2:
            col_sc1, col_sc2 = st.columns([2, 1])
            with col_sc1:
                fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Réalité', 'y': 'Prédiction'}, opacity=0.6, trendline="ols", trendline_color_override="red")
                fig_scatter.add_shape(type="line", line=dict(dash='dash', color='grey'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col_sc2:
                st.info("Alignement parfait = Ligne grise.")

        with tab3:
            residuals = y_test - y_pred
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                fig_hist = px.histogram(residuals, nbins=30, labels={'value': 'Erreur'}, color_discrete_sequence=['#ef553b'])
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_res2:
                fig_res_time = px.scatter(x=dates_test, y=residuals, labels={'x': 'Date', 'y': 'Erreur'})
                fig_res_time.add_hline(y=0, line_dash="dash", line_color="green")
                st.plotly_chart(fig_res_time, use_container_width=True)

        with tab4:
            importances = model.feature_importances_
            df_imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Pas assez de données pour l'IA.")
else:
    st.error("CSV introuvable.")
