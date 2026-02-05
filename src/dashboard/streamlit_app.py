import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import psycopg
from dotenv import load_dotenv
import sys
import google.generativeai as genai
import io
from PIL import Image

# --- 0. AI Configuration & Helper Function ---

def configure_genai():
    """Configures the Google Gemini API."""
    load_dotenv()
    api_key = os.environ.get('GEMINI_API_TOKEN')
    if not api_key:
        st.sidebar.error("⚠️ GEMINI_API_TOKEN not found in .env")
        return False
    
    genai.configure(api_key=api_key)
    return True

def analyze_chart_with_gemini(fig, chart_description):
    """
    Converts a Matplotlib figure to an image, sends it to Gemini,
    and returns the analysis.
    """
    # 1. Convert Plot to Image in Memory
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)

    # 2. Define the prompt
    prompt = f"""
    You are an expert environmental data analyst. 
    Analyze the attached chart titled: '{chart_description}'.
    
    Please provide:
    1. A brief summary of the visible trends.
    2. Any significant outliers or anomalies (e.g., very high pollution spikes).
    3. A potential explanation or actionable insight based on this data.
    
    Keep the response concise and easy to read.
    """

    # 3. Call Gemini (Using 1.5 Flash for speed/efficiency)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        with st.spinner("✨ Gemini is analyzing the chart..."):
            response = model.generate_content([prompt, image])
            return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

def render_ai_analysis_ui(fig, title):
    """
    Helper to render the Streamlit UI for AI analysis.
    """
    st.markdown("##### 🤖 AI Insights")
    
    # We use an expander so it doesn't clutter the view immediately
    with st.expander(f"Analyze '{title}' with Gemini"):
        # We use a button so we don't burn API credits automatically on every page reload
        if st.button(f"Generate Analysis for {title}", key=title):
            if configure_genai():
                analysis = analyze_chart_with_gemini(fig, title)
                st.markdown(analysis)
            else:
                st.warning("Please set your GEMINI_API_TOKEN to use this feature.")

# --- 1. Database Loading Function ---

@st.cache_data(ttl=900) 
def load_data_from_db():
    load_dotenv()
    db_url = os.environ.get('POSTGRES_URL')

    if not db_url:
        st.error("Error: POSTGRES_URL environment variable not found.")
        return None

    query = "SELECT * FROM aqi_data ORDER BY timestamp;"

    try:
        # st.info("Connecting to Vercel Postgres database...") # Optional: Comment out to reduce UI clutter
        with psycopg.connect(db_url) as conn:
            df = pd.read_sql(query, conn)
        return df

    except Exception as e:
        st.error(f"An error occurred while connecting to the database: {e}")
        return None

# --- 2. Preprocessing Function ---

@st.cache_data
def preprocess_data(df_raw):
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=7)
    
    numeric_cols = ['aqi', 'pm25', 'pm10', 'co', 'so2', 'no2', 'o3']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            location_mean = df.groupby('location')[col].transform('mean')
            df[col] = df[col].fillna(location_mean)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date
    
    return df

# --- 3. Plotting Functions (Updated with AI Integration) ---

def plot_avg_by_location(df, column, title, palette="viridis"):
    if column not in df.columns:
        st.warning(f"Cannot plot '{column}': column not found.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted size for better AI reading
    avg_data = df.groupby('location')[column].mean().sort_values()
    sns.barplot(x=avg_data.values, y=avg_data.index, palette=palette, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f'Average {column.upper()}', fontsize=12)
    ax.set_ylabel('Location', fontsize=12)
    plt.tight_layout()
    
    st.pyplot(fig)
    render_ai_analysis_ui(fig, title) # <--- AI INTEGRATION

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    pollutant_cols = ['aqi', 'pm25', 'pm10', 'co', 'so2', 'no2', 'o3']
    existing_cols = [col for col in pollutant_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if 'aqi' not in existing_cols:
        return
        
    correlation_matrix = df[existing_cols].corr()
    aqi_correlations = correlation_matrix['aqi'].sort_values(ascending=False)

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of AQI and Pollutants', fontsize=16)
    
    st.pyplot(fig)
    
    st.markdown("**Correlation with AQI:**")
    st.dataframe(aqi_correlations)
    
    render_ai_analysis_ui(fig, "Correlation Heatmap of Pollutants") # <--- AI INTEGRATION

def plot_pollution_profile(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    pollutant_cols = ['pm25', 'pm10', 'co', 'so2', 'no2', 'o3']
    existing_cols = [col for col in pollutant_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not existing_cols:
        return

    profile_df = df.groupby('location')[existing_cols].mean()
    normalized_profile = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min())
    profile_percent = normalized_profile.apply(lambda x: x / x.sum(), axis=1).fillna(0)
    
    sort_col = existing_cols[0]
    profile_percent_sorted = profile_percent.sort_values(by=sort_col, ascending=False)
    
    profile_percent_sorted.plot(
        kind='barh', stacked=True, cmap='tab10', width=0.8, ax=ax
    )
    
    ax.set_title('Normalized Pollution Profile by Location', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Pollutants")
    plt.tight_layout()
    
    st.pyplot(fig)
    render_ai_analysis_ui(fig, "Pollution Profile Composition") # <--- AI INTEGRATION

# --- Location-Specific Plots ---

def plot_daily_aqi(location_df, location_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_avg = location_df.groupby('date')['aqi'].mean()
    daily_avg.plot(kind='line', marker='.', linestyle='-', ax=ax, alpha=0.7)
    ax.set_title(f'Average Daily AQI in {location_name}', fontsize=16)
    ax.set_ylabel('Average AQI')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    st.pyplot(fig)
    render_ai_analysis_ui(fig, f"Daily AQI Trend for {location_name}") # <--- AI INTEGRATION
        
def plot_hourly_avg(location_df, location_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_avg = location_df.groupby('hour')['aqi'].mean()
    # hourly_avg = hourly_avg.reindex(range(24), fill_value=None) 
    
    sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette="plasma")
    plt.title(f'Average AQI by Hour of Day in {location_name} (GMT+7)', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average AQI', fontsize=12)
    plt.tight_layout()
    
    st.pyplot(fig)
    render_ai_analysis_ui(fig, f"Hourly AQI Profile for {location_name}") # <--- AI INTEGRATION
        
def plot_dow_avg(location_df, location_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = location_df.groupby('day_of_week')['aqi'].mean().reindex(days_order)
    
    sns.barplot(x=dow_avg.index, y=dow_avg.values, palette="cividis")
    plt.title(f'Average AQI by Day of Week in {location_name}', fontsize=16)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Average AQI', fontsize=12)
    plt.tight_layout()
    
    st.pyplot(fig)
    render_ai_analysis_ui(fig, f"Weekly AQI Profile for {location_name}") # <--- AI INTEGRATION

# --- 4. Main App Logic ---

def main():
    st.set_page_config(layout="wide", page_title="AQI AI Dashboard")
    
    st.title("🏭 AQI Analysis Dashboard")

    df_raw = load_data_from_db()

    if df_raw is None or df_raw.empty:
        st.warning("Failed to load data.")
        return

    df = preprocess_data(df_raw)

    # Sidebar
    st.sidebar.title("Navigation")
    locations = sorted(df['location'].unique())
    page = st.sidebar.selectbox("Select Dashboard", ["Overall Dashboard"] + locations)
    
    # Check for API key immediately
    if not os.environ.get("GEMINI_API_TOKEN"):
        st.sidebar.warning("⚠️ No GEMINI_API_TOKEN detected in .env")

    if page == "Overall Dashboard":
        st.header("Overall Dashboard")
        
        # KPIs
        try:
            avg_aqi_all = df['aqi'].mean()
            max_aqi_loc = df.groupby('location')['aqi'].mean().idxmax()
            max_aqi_val = df.groupby('location')['aqi'].mean().max()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Locations", len(locations))
            col2.metric("Global Avg AQI", f"{avg_aqi_all:.2f}")
            col3.metric("Most Polluted", f"{max_aqi_loc} ({max_aqi_val:.2f})")
        except:
            pass
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Correlations")
            plot_correlation_heatmap(df)
        with col2:
            st.subheader("Pollution Profiles")
            plot_pollution_profile(df)
            
        st.divider()

        st.subheader("Pollutant Levels by Location")
        pollutant_cols = ['aqi', 'pm25', 'pm10', 'co', 'so2', 'no2', 'o3']
        existing_cols = [col for col in pollutant_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if existing_cols:
            tabs = st.tabs([col.upper() for col in existing_cols])
            for i, col_name in enumerate(existing_cols):
                with tabs[i]:
                    plot_avg_by_location(df, col_name, f'Average {col_name.upper()} by Location')

    else: 
        st.header(f"📍 Dashboard: {page}")
        location_df = df[df['location'] == page].copy()
        
        if location_df.empty:
            st.warning(f"No data for {page}")
            return
            
        st.subheader("Key Metrics")
        try:
            loc_avg = location_df['aqi'].mean()
            loc_latest = location_df.sort_values('timestamp', ascending=False)['aqi'].iloc[0]
            loc_max = location_df['aqi'].max()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average AQI", f"{loc_avg:.2f}")
            col2.metric("Current AQI", f"{loc_latest:.0f}")
            col3.metric("Peak AQI", f"{loc_max:.0f}")
        except:
            pass
        
        st.divider()

        st.subheader("Trends")
        plot_daily_aqi(location_df, page)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            plot_hourly_avg(location_df, page)
        with col2:
            plot_dow_avg(location_df, page)

if __name__ == "__main__":
    main()