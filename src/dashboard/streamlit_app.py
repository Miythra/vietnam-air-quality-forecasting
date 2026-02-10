import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Vietnam Air Quality AI - Analytics",
    page_icon="📊",
    layout="wide"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
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
    """Charge et nettoie les données historiques."""
    # Liste des chemins possibles
    possible_paths = [
        "data/aqi_data.csv", "src/data/aqi_data.csv", "aqi_data.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except: continue
            
    if df is not None:
        # Conversion Date
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        
        # Conversion Numérique
        cols_num = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna().sort_values('timestamp')
    return None

# --- ENTRAÎNEMENT DU MODÈLE (CACHÉ) ---
@st.cache_resource
def train_model(df_loc):
    """Entraîne le modèle et renvoie les résultats pour visualisation."""
    features = ['pm25', 'no2', 'so2', 'co', 'o3']
    # Vérification des colonnes existantes
    features = [f for f in features if f in df_loc.columns]
    
    if len(df_loc) < 20 or not features:
        return None, None, None, None, None

    X = df_loc[features]
    y = df_loc['aqi']
    
    # Split Chronologique (pas aléatoire) pour respecter le temps
    # On garde les 20% les plus récents pour le test
    split_idx = int(len(df_loc) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df_loc['timestamp'].iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, dates_test

# --- INTERFACE PRINCIPALE ---
st.title("📊 Analyse Avancée & Performance IA")
st.markdown("Comparaison interactive entre la **Réalité (Ground Truth)** et les **Prédictions IA** sur les données historiques.")

df = load_data()

if df is not None:
    # --- SIDEBAR : FILTRES GLOBAUX ---
    st.sidebar.header("🎛️ Configuration")
    
    # 1. Choix Ville
    locations = sorted(df['location'].unique())
    selected_location = st.sidebar.selectbox("📍 Ville cible", locations)
    
    # Filtrage Données
    df_loc = df[df['location'] == selected_location]
    
    # 2. Entraînement
    with st.spinner(f"Entraînement du modèle pour {selected_location}..."):
        model, X_test, y_test, y_pred, dates_test = train_model(df_loc)

    if model is not None:
        # Calcul Métriques
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # --- BLOC KPI (MÉTRIQUES) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Échantillons Testés", len(y_test), border=True)
        col2.metric("Précision (R²)", f"{r2:.2%}", delta_color="normal" if r2 > 0.7 else "inverse", border=True)
        col3.metric("Erreur Moyenne (MAE)", f"{mae:.1f}", delta="-Low is good", delta_color="inverse", border=True)
        col4.metric("RMSE (Erreur Quadratique)", f"{rmse:.1f}", border=True)

        st.markdown("---")

        # --- ONGLETS DE VISUALISATION ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Analyse Temporelle", 
            "🎯 Précision & Corrélation", 
            "📉 Analyse des Erreurs",
            "🧠 Intégration Modèle"
        ])

        # === TAB 1 : SÉRIE TEMPORELLE ===
        with tab1:
            st.subheader("Réalité vs Prédiction au fil du temps")
            
            fig_ts = go.Figure()
            # Réalité
            fig_ts.add_trace(go.Scatter(
                x=dates_test, y=y_test, 
                mode='lines', name='Réalité (Mesuré)',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{y:.0f} AQI<extra></extra>'
            ))
            # Prédiction
            fig_ts.add_trace(go.Scatter(
                x=dates_test, y=y_pred, 
                mode='lines', name='Prédiction IA',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                hovertemplate='%{y:.0f} AQI<extra></extra>'
            ))
            
            fig_ts.update_layout(
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="AQI",
                legend=dict(orientation="h", y=1.1),
                height=500
            )
            # Ajout Range Slider
            fig_ts.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_ts, use_container_width=True)

        # === TAB 2 : SCATTER PLOT ===
        with tab2:
            col_sc1, col_sc2 = st.columns([2, 1])
            with col_sc1:
                st.subheader("Nuage de points : Prédit vs Réel")
                st.caption("Un modèle parfait alignerait tous les points sur la ligne rouge diagonale.")
                
                fig_scatter = px.scatter(
                    x=y_test, y=y_pred, 
                    labels={'x': 'Valeur Réelle (AQI)', 'y': 'Valeur Prédite (AQI)'},
                    opacity=0.6,
                    trendline="ols", # Ligne de tendance
                    trendline_color_override="red"
                )
                # Ligne parfaite y=x
                fig_scatter.add_shape(
                    type="line", line=dict(dash='dash', color='grey'),
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max()
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col_sc2:
                st.info("""
                **Comment lire ce graphe ?**
                * **Sur la ligne grise :** Prédiction parfaite.
                * **Au-dessus :** L'IA surestime la pollution.
                * **En-dessous :** L'IA sous-estime la pollution.
                """)

        # === TAB 3 : ERREURS (RESIDUALS) ===
        with tab3:
            st.subheader("Distribution des Erreurs (Résidus)")
            
            residuals = y_test - y_pred
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("**1. Histogramme des erreurs**")
                fig_hist = px.histogram(
                    residuals, nbins=30, 
                    labels={'value': 'Erreur (Réel - Prédit)'},
                    color_discrete_sequence=['#ef553b']
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_res2:
                st.markdown("**2. Erreur au fil du temps**")
                fig_res_time = px.scatter(
                    x=dates_test, y=residuals,
                    labels={'x': 'Date', 'y': 'Erreur (Residual)'}
                )
                # Ligne zéro
                fig_res_time.add_hline(y=0, line_dash="dash", line_color="green")
                st.plotly_chart(fig_res_time, use_container_width=True)

        # === TAB 4 : IMPORTANCE DES FEATURES ===
        with tab4:
            st.subheader("Qu'est-ce qui influence le plus l'IA ?")
            
            # Extraction importance
            importances = model.feature_importances_
            feature_names = X_test.columns
            df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            df_imp = df_imp.sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(
                df_imp, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("""
            > **Note :** Ce graphique montre quels polluants (PM2.5, NO2, etc.) ont le plus de poids dans la décision de l'IA pour calculer l'AQI global.
            """)

    else:
        st.warning("Pas assez de données pour entraîner le modèle sur cette ville.")
else:
    st.error("⚠️ Fichier CSV introuvable. Veuillez vérifier vos données.")
