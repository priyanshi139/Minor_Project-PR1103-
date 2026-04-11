import streamlit as st
import pandas as pd
import plotly.express as px


# PAGE CONFIGURATION


st.set_page_config(layout="wide")
st.title("AI Based Air Quality and Health Risk Monitoring Dashboard for India")

st.markdown("""
This dashboard analyzes **Air Quality Index (AQI), weather conditions, and health risk levels**
across different cities in India. The system uses environmental parameters to estimate
health risk levels and provides visual insights into pollution trends.
""")


# DATA LOADING FUNCTION


@st.cache_data
def load_data():

    # Load dataset
    df = pd.read_csv(r"C:\Users\Priya\Downloads\INDIA_AQI_CLEANED.csv")

    # Convert datetime column
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Remove rows with missing AQI
    df = df.dropna(subset=["US_AQI"])

    # Weather columns used for health risk calculation
    weather_cols = [
        "Solar_Radiation_Wm2",
        "Rain_mm",
        "Temp_2m_C",
        "Humidity_Percent",
        "Wind_Speed_10m_kmh",
        "Surface_Pressure_hPa"
    ]

    # Replace missing values using mean
    for col in weather_cols:
        mean_val = df[col].replace(0, pd.NA).mean()
        df[col] = df[col].replace(0, mean_val)
        df[col] = df[col].fillna(mean_val)

    # HEALTH RISK SCORE CALCULATION

    df["Health_Risk_Score"] = (
        df["US_AQI"] * 0.5 +
        df["Temp_2m_C"] * 0.1 +
        df["Humidity_Percent"] * 0.1 +
        df["Wind_Speed_10m_kmh"] * 0.1 +
        df["Solar_Radiation_Wm2"] * 0.1 +
        df["Surface_Pressure_hPa"] * 0.05 +
        df["Rain_mm"] * 0.05
    )

    # Function to classify health risk
    def classify(score):
        if score <= 50:
            return "Low Risk"
        elif score <= 100:
            return "Moderate Risk"
        elif score <= 200:
            return "High Risk"
        else:
            return "Severe Risk"

    df["Health_Risk_Level"] = df["Health_Risk_Score"].apply(classify)

    # Extract hour for time-of-day analysis
    df["Hour"] = df["Datetime"].dt.hour

    return df


df = load_data()



aqi_colors = {
    "Good":                "green",      # AQI 0–50    → green
    "Moderate":            "yellow",     # AQI 51–100  → yellow
    "Unhealthy_Sensitive": "orange",     # AQI 101–150 → orange (was peach before — fixed)
    "Unhealthy":           "red",        # AQI 151–200 → red (darker than orange — correct order)
    "Very_Unhealthy":      "purple",     # AQI 201–300 → purple
    "Hazardous":           "black",      # AQI 301–500 → black (darkest — worst)
}

# Category order from best to worst 
category_order = [
    "Good",
    "Moderate",
    "Unhealthy_Sensitive",
    "Unhealthy",
    "Very_Unhealthy",
    "Hazardous"
]


# SIDEBAR FILTERS


st.sidebar.header("Dashboard Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

selected_city = st.sidebar.selectbox(
    "Select City",
    sorted(df["City"].unique())
)

filtered_df = df[df["Year"] == selected_year]

city_df = filtered_df[filtered_df["City"] == selected_city]

city_df = city_df.sort_values("Datetime")

latest = city_df.iloc[-1]


# KEY PERFORMANCE INDICATORS


st.header("Air Quality Key Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average AQI",
    round(filtered_df["US_AQI"].mean(), 2)
)

col2.metric(
    "Most Polluted City",
    filtered_df.groupby("City")["US_AQI"].mean().idxmax()
)

col3.metric(
    "Dominant AQI Category",
    filtered_df["AQI_Category"].mode()[0]
)

col4.metric(
    "Average Health Risk Score",
    round(filtered_df["Health_Risk_Score"].mean(), 2)
)

st.markdown("---")


# INDIA AQI MAP — FIXED
# hover_data added so click/hover shows correct category name
# category_orders ensures legend is sorted Good → Hazardous


st.header("Geographical Distribution of AQI Across Cities")

map_data = filtered_df.groupby(
    ["City", "Latitude", "Longitude", "AQI_Category"]
)["US_AQI"].mean().reset_index()

fig_map = px.scatter_geo(
    map_data,
    lat="Latitude",
    lon="Longitude",
    hover_name="City",
    hover_data={
        "US_AQI": ":.2f",           # show AQI value on hover
        "AQI_Category": True,        # show correct category on hover
        "Latitude": False,           # hide lat/lon from hover
        "Longitude": False,
    },
    size="US_AQI",
    size_max=20,
    color="AQI_Category",
    color_discrete_map=aqi_colors,
    category_orders={"AQI_Category": category_order},  
    scope="asia"
)

fig_map.update_layout(
    legend_title_text="AQI Category",
    legend=dict(
        itemsizing="constant",
        traceorder="normal"
    )
)

st.plotly_chart(fig_map, use_container_width=True)


# HEALTH RISK PREDICTION PANEL


st.header("Current Health Risk Prediction Based on AQI and Weather")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Predicted Health Risk Level")

    st.error(f"{latest['Health_Risk_Level']}")

with col2:

    st.subheader("Latest Environmental Conditions")

    st.write(f"AQI Level: {latest['US_AQI']}")
    st.write(f"Temperature (°C): {latest['Temp_2m_C']}")
    st.write(f"Humidity (%): {latest['Humidity_Percent']}")
    st.write(f"Wind Speed (km/h): {latest['Wind_Speed_10m_kmh']}")
    st.write(f"Solar Radiation (W/m²): {round(latest['Solar_Radiation_Wm2'], 2)}")
    st.write(f"Rainfall (mm): {round(latest['Rain_mm'], 2)}")


# HEALTH PRECAUTIONS


st.subheader("Suggested Health Precautions")

risk = latest["Health_Risk_Level"]

if risk == "Low Risk":
    st.success("Air quality is good. Outdoor activities are safe.")
elif risk == "Moderate Risk":
    st.warning("Sensitive individuals should limit prolonged outdoor exposure.")
elif risk == "High Risk":
    st.error("Wear a mask and avoid outdoor activities if possible.")
else:
    st.error("Stay indoors and avoid outdoor exposure.")


# AQI TREND OVER TIME


st.header("AQI Trend Over Time for Selected City")

fig = px.line(
    city_df,
    x="Datetime",
    y="US_AQI",
    color="AQI_Category",
    color_discrete_map=aqi_colors,
    category_orders={"AQI_Category": category_order}
)

st.plotly_chart(fig, use_container_width=True)


# WIND SPEED ANALYSIS


st.header("Wind Speed vs AQI Relationship")

fig = px.scatter(
    filtered_df,
    x="Wind_Speed_10m_kmh",
    y="US_AQI",
    color="AQI_Category",
    color_discrete_map=aqi_colors,
    category_orders={"AQI_Category": category_order}
)

st.plotly_chart(fig, use_container_width=True)


# SUNSHINE IMPACT


st.header("Impact of Solar Radiation on Air Quality")

fig = px.scatter(
    filtered_df,
    x="Solar_Radiation_Wm2",
    y="US_AQI",
    color="AQI_Category",
    color_discrete_map=aqi_colors,
    category_orders={"AQI_Category": category_order}
)

st.plotly_chart(fig, use_container_width=True)


# TIME OF DAY ANALYSIS


st.header("Average AQI Variation During Different Hours of the Day")

hourly = filtered_df.groupby("Hour")["US_AQI"].mean().reset_index()

fig = px.line(
    hourly,
    x="Hour",
    y="US_AQI",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)


# TOP POLLUTED CITIES


st.header("Top 10 Most Polluted Cities")

city_avg = filtered_df.groupby("City")["US_AQI"].mean()

top10 = city_avg.sort_values(ascending=False).head(10)

fig = px.bar(
    top10,
    labels={"value": "Average AQI"},
)

st.plotly_chart(fig, use_container_width=True)


# POLLUTANT COMPARISON


st.header("Average Pollutant Concentration Levels")

pollutants = filtered_df[
    ["PM2_5_ugm3", "PM10_ugm3", "NO2_ugm3", "SO2_ugm3"]
].mean()

fig = px.bar(
    pollutants,
    labels={"value": "Average Concentration"}
)

st.plotly_chart(fig, use_container_width=True)


# FESTIVAL IMPACT ANALYSIS


st.header("Impact of Festival Periods on Air Quality")

festival = filtered_df.groupby("Festival_Period")["US_AQI"].mean()

fig = px.bar(festival)

st.plotly_chart(fig, use_container_width=True)


# CROP BURNING IMPACT


st.header("Impact of Crop Burning Season on AQI")

crop = filtered_df.groupby("Crop_Burning_Season")["US_AQI"].mean()

fig = px.bar(crop)

st.plotly_chart(fig, use_container_width=True)


# CITY RANKING TABLE


st.header("City AQI Ranking Table")

ranking = city_avg.sort_values(ascending=False)

st.dataframe(ranking)

st.success("Stay informed about air quality levels and take precautions to protect your health.")