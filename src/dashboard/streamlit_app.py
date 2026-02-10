import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vietnam Air Quality AI - Technical Showcase",
    page_icon="🤖",
    layout="wide"
)

# --- CITY COORDINATES (Based on your data) ---
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

# --- CUSTOM CSS FOR EXPLANATIONS ---
st.markdown("""
<style>
    .explanation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 15px;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
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
        # Standardize Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        
        # Ensure numbers are floats
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

# --- MODEL TRAINING ---
@st.cache_resource
def train_model(df_loc):
    features = ['pm25', 'no2', 'so2', 'co', 'o3']
    features = [f for f in features if f in df_loc.columns]
    
    if len(df_loc) < 20 or not features:
        return None, None, None, None, None

    X = df_loc[features]
    y = df_loc['aqi']
    
    # Chronological Split (Train on past, Test on future)
    split_idx = int(len(df_loc) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df_loc['timestamp'].iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, dates_test

# --- MAIN APP ---
st.title("🤖 Vietnam Air Quality AI - Performance Showcase")
st.markdown("""
**Welcome.** This dashboard demonstrates the capabilities of a Machine Learning model (Random Forest) designed to predict Air Quality in Vietnam.
It highlights the **accuracy**, **reliability**, and **decision-making logic** of the artificial intelligence.
""")

df = load_data()

if df is not None:
    # --- CONFIGURATION SIDEBAR ---
    st.sidebar.header("⚙️ Configuration")
    locations = sorted(df['location'].unique())
    selected_location = st.sidebar.selectbox("📍 Select a Target City", locations)
    
    # --- 1. GEOGRAPHICAL OVERVIEW ---
    st.subheader("1. Data Scope & Geographic Coverage")
    st.markdown("""
    <div class="explanation-box">
    <b>What are we looking at?</b><br>
    This map shows the most recent data points collected. It automatically highlights the <b>Target City</b> you selected, 
    while also benchmarking it against the cleanest (✅) and most polluted (❌) cities currently in the dataset.
    </div>
    """, unsafe_allow_html=True)
    
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
                    'size': 15 if label_type == 'Target' else 10,
                    'color': get_aqi_color(row['aqi'])
                })

        # Add points logic
        if not row_target.empty: add_point(row_target.iloc[0], 'Target City')
        if row_best['location'] != selected_location: add_point(row_best, 'Best AQI')
        if row_worst['location'] != selected_location: add_point(row_worst, 'Worst AQI')
        
        # Fallback: Add all other cities as small dots for context
        for idx, row in daily_stats.iterrows():
            if row['location'] not in [selected_location, row_best['location'], row_worst['location']]:
                lat, lon = CITY_COORDS.get(row['location'], [None, None])
                if lat:
                    map_points.append({'location': row['location'], 'aqi': int(row['aqi']), 'lat': lat, 'lon': lon, 'type': 'Others', 'size': 6, 'color': 'grey'})

        df_map = pd.DataFrame(map_points)
        
        if not df_map.empty:
            fig_map = px.scatter_mapbox(
                df_map, lat="lat", lon="lon", color="type", size="size",
                hover_name="location", hover_data={"aqi": True, "lat": False, "lon": False, "size": False},
                zoom=5, center={"lat": 16.0, "lon": 106.0},
                mapbox_style="carto-positron", title=f"Situation Snapshot ({latest_date})"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Context Metrics
            m1, m2, m3 = st.columns(3)
            if not row_target.empty:
                m1.metric(f"📍 Selected: {selected_location}", f"{int(row_target.iloc[0]['aqi'])} AQI", "Current Level")
            m2.metric(f"✅ Cleanest: {row_best['location']}", f"{int(row_best['aqi'])} AQI", "Best in Vietnam")
            m3.metric(f"❌ Highest Pollution: {row_worst['location']}", f"{int(row_worst['aqi'])} AQI", "Worst in Vietnam", delta_color="inverse")
        else:
            st.warning("⚠️ Map data unavailable. Please check coordinate mapping.")

    st.divider()

    # --- 2. AI PERFORMANCE ANALYSIS ---
    st.subheader(f"2. AI Model Performance for {selected_location}")
    
    # Train Model
    df_loc = df[df['location'] == selected_location]
    with st.spinner(f"Training Machine Learning Model for {selected_location}..."):
        model, X_test, y_test, y_pred, dates_test = train_model(df_loc)

    if model is not None:
        # Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # --- EXPLANATION BOX FOR METRICS ---
        st.markdown("""
        <div class="explanation-box">
        <b>How to interpret the AI Scorecard below:</b><br>
        <ul>
            <li><b>R² Score (Accuracy):</b> Think of this as a test grade out of 100%. A score above 80% means the AI understands the pollution patterns very well.</li>
            <li><b>Mean Absolute Error (MAE):</b> This is the "Average Mistake". If the real AQI is 150 and the AI guesses 155, the error is 5. Lower is better.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("🧪 Test Samples", len(y_test), "Data points used for testing")
        col2.metric("🎯 Model Accuracy (R²)", f"{r2:.1%}", "Higher is better")
        col3.metric("📉 Average Error (MAE)", f"{mae:.1f}", "Lower is better", delta_color="inverse")

        st.markdown("---")

        # --- TABS FOR VISUALIZATION ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Visual Proof (Time Series)", 
            "🎯 Accuracy Check (Scatter)", 
            "🔎 Quality Control (Errors)", 
            "🧠 The AI's Logic (Drivers)"
        ])

        # === TAB 1: TIME SERIES ===
        with tab1:
            st.subheader("Does the AI predict the spikes?")
            st.markdown("""
            **How to read this:**
            * **Blue Line:** What actually happened (Real data).
            * **Orange Dotted Line:** What the AI predicted.
            * **Goal:** The orange line should stick as close as possible to the blue line. If they overlap, the AI is working perfectly.
            """)
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Reality (Actual)', line=dict(color='#1f77b4', width=2)))
            fig_ts.add_trace(go.Scatter(x=dates_test, y=y_pred, mode='lines', name='AI Prediction', line=dict(color='#ff7f0e', width=2, dash='dot')))
            fig_ts.update_layout(hovermode="x unified", xaxis_title="Time", yaxis_title="AQI Level", legend=dict(orientation="h", y=1.1), height=500)
            st.plotly_chart(fig_ts, use_container_width=True)

        # === TAB 2: SCATTER ===
        with tab2:
            st.subheader("Alignment Verification")
            st.markdown("""
            **How to read this:**
            This chart compares every single prediction against reality.
            * **Grey Line (Diagonal):** The "Perfect Prediction" line.
            * **Dots:** The AI's guesses.
            * **Goal:** We want all dots to be tightly packed around the grey diagonal line. Dots far away represent "misses".
            """)
            
            col_sc1, col_sc2 = st.columns([3, 1])
            with col_sc1:
                fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real AQI', 'y': 'Predicted AQI'}, opacity=0.6, trendline="ols", trendline_color_override="red")
                fig_scatter.add_shape(type="line", line=dict(dash='dash', color='grey'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                st.plotly_chart(fig_scatter, use_container_width=True)

        # === TAB 3: RESIDUALS ===
        with tab3:
            st.subheader("Error Pattern Analysis")
            st.markdown("""
            **How to read this:**
            This checks if the AI has a "bias" (e.g., does it always overestimate?).
            * **Center at 0:** Ideally, the histogram should look like a bell curve centered at 0.
            * **Meaning:** This implies the AI's errors are random and not due to a systematic flaw in the code.
            """)
            
            residuals = y_test - y_pred
            fig_hist = px.histogram(residuals, nbins=30, labels={'value': 'Prediction Error (Real - Predicted)'}, color_discrete_sequence=['#ef553b'])
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        # === TAB 4: FEATURE IMPORTANCE ===
        with tab4:
            st.subheader("What drives the AI's decisions?")
            st.markdown("""
            **How to read this:**
            The AI analyzes multiple gases (PM2.5, NO2, CO...) to calculate the Air Quality Index.
            * **Longer Bar:** Means this specific pollutant is the **main driver** of pollution in this city.
            * **Insight:** This tells us exactly which gas we should monitor most closely.
            """)
            
            importances = model.feature_importances_
            df_imp = pd.DataFrame({'Pollutant': X_test.columns, 'Importance': importances}).sort_values('Importance', ascending=True)
            # Rename for clarity
            df_imp['Pollutant'] = df_imp['Pollutant'].str.upper()
            
            fig_imp = px.bar(df_imp, x='Importance', y='Pollutant', orientation='h', color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Not enough data points to train the AI model for this city yet.")
else:
    st.error("Data file not found. Please ensure the CSV is in the repository.")
