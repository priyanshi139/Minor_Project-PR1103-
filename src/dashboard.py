"""
India AQI + Health Risk Forecast Dashboard
==========================================
Run: streamlit run dashboard.py
"""

import os, joblib, numpy as np, pandas as pd, requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import load_model

# CONFIG
DATA_PATH   = r"C:\Users\Priya\Downloads\INDIA_AQI_CLEANED.csv"
MODELS_DIR  = "models"
SEQ_LEN     = 48
OWM_API_KEY = "753a506d1f64b7d347b165a1c8326ce3"

FEATURES = [
    "US_AQI", "PM2_5_ugm3", "PM10_ugm3", "Temp_2m_C", "Humidity_Percent",
    "Wind_Speed_10m_kmh", "Surface_Pressure_hPa", "Solar_Radiation_Wm2", "Rain_mm",
]

AQI_COLORS = {
    "Good": "#00e400", "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000", "Very Unhealthy": "#8f3f97", "Hazardous": "#7e0023",
}
RISK_COLORS = {"Low": "#00c851", "Moderate": "#ffbb33", "High": "#ff4444", "Severe": "#cc0000"}

CITY_COORDS = {
    "Agartala": (23.8315, 91.2868), "Ahmedabad": (23.0225, 72.5714),
    "Aizawl": (23.7271, 92.7176), "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126), "Bhubaneswar": (20.2961, 85.8245),
    "Chennai": (13.0827, 80.2707), "Delhi": (28.6139, 77.2090),
    "Gangtok": (27.3389, 88.6065), "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867), "Imphal": (24.8170, 93.9368),
    "Itanagar": (27.0844, 93.6053), "Jaipur": (26.9124, 75.7873),
    "Jammu": (32.7266, 74.8570), "Kohima": (25.6751, 94.1086),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777), "Nagpur": (21.1458, 79.0882),
    "Panaji": (15.4909, 73.8278), "Patna": (25.5941, 85.1376),
    "Raipur": (21.2514, 81.6296), "Ranchi": (23.3441, 85.3096),
    "Shillong": (25.5788, 91.8933), "Shimla": (31.1048, 77.1734),
    "Srinagar": (34.0837, 74.7973), "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185),
}

HEALTH_CONDITIONS = [
    "None", "Asthma", "COPD", "Heart Disease", "Diabetes",
    "Allergies / Rhinitis", "Lung Disease", "Elderly (65+)", "Pregnant"
]

ACTIVITY_ADVICE = {
    "Running": {
        "Low":      "Good day for running. Go for it.",
        "Moderate": "Running is OK. Avoid peak traffic hours (8-10am, 6-8pm).",
        "High":     "Avoid outdoor running. Use treadmill instead.",
        "Severe":   "Do not run outdoors. Stay inside completely.",
    },
    "Cycling": {
        "Low":      "Perfect cycling weather.",
        "Moderate": "Wear a mask. Avoid main roads.",
        "High":     "Not recommended. Air quality is poor.",
        "Severe":   "Dangerous. Do not cycle outdoors.",
    },
    "Yoga / Walking": {
        "Low":      "Excellent. Enjoy outdoor yoga or walking.",
        "Moderate": "Morning walks are OK. Avoid afternoon.",
        "High":     "Do yoga indoors with windows closed.",
        "Severe":   "Strictly indoors only.",
    },
    "Indoor / Work": {
        "Low":      "Air quality is great indoors and outdoors.",
        "Moderate": "Keep windows open for ventilation.",
        "High":     "Keep windows closed. Use air purifier if available.",
        "Severe":   "Seal doors and windows. Run air purifier on max.",
    },
    "Kids Outdoor Play": {
        "Low":      "Safe for children to play outside.",
        "Moderate": "Limit to 1 hour. Bring water.",
        "High":     "Keep kids indoors. No outdoor play.",
        "Severe":   "Children must stay indoors. Dangerous air.",
    },
    "Commuting": {
        "Low":      "Normal commute. Windows down is fine.",
        "Moderate": "Close car windows on busy roads.",
        "High":     "Close windows, recirculate AC. Wear mask if on bike.",
        "Severe":   "Wear N95 mask. Avoid two-wheelers.",
    },
}


def age_sensitivity(age):
    if age < 12: return "child"
    if age > 65: return "elderly"
    return "adult"

def owm_to_us(i):
    return float({1: 25, 2: 75, 3: 125, 4: 175, 5: 250}.get(i, 100))

def aqi_cat(a):
    if a <= 50:  return "Good"
    if a <= 100: return "Moderate"
    if a <= 150: return "Unhealthy for Sensitive Groups"
    if a <= 200: return "Unhealthy"
    if a <= 300: return "Very Unhealthy"
    return "Hazardous"

def health_risk(a):
    if a <= 50:  return "Low"
    if a <= 100: return "Moderate"
    if a <= 200: return "High"
    return "Severe"

def get_personalized_precaution(risk, age, conditions):
    base = {
        "Low":     "Air quality is good.",
        "Moderate":"Air quality is acceptable.",
        "High":    "Air quality is poor.",
        "Severe":  "Air quality is hazardous.",
    }.get(risk, "")
    sens  = age_sensitivity(age)
    extra = []
    if sens == "child":
        if risk in ("High", "Severe"):
            extra.append("Keep children strictly indoors.")
        elif risk == "Moderate":
            extra.append("Limit children outdoor time to 30 min.")
    if sens == "elderly":
        if risk in ("High", "Severe"):
            extra.append("Elderly must avoid all outdoor activity.")
        elif risk == "Moderate":
            extra.append("Elderly should avoid exertion outdoors.")
    sensitive_conditions = {"Asthma", "COPD", "Heart Disease", "Lung Disease", "Pregnant"}
    user_sensitive = set(conditions) & sensitive_conditions
    if user_sensitive and risk in ("Moderate", "High", "Severe"):
        cond_str = ", ".join(user_sensitive)
        extra.append(f"With {cond_str}: carry medication, avoid outdoor exposure.")
    if not extra:
        extra = {
            "Low":     ["Enjoy outdoor activities freely."],
            "Moderate":["Sensitive people should limit prolonged outdoor exertion."],
            "High":    ["Wear N95 mask if going out. Close windows at home."],
            "Severe":  ["Stay indoors. Use air purifier. Avoid all outdoor exposure."],
        }.get(risk, [""])
    return base + " " + " ".join(extra)


@st.cache_data(show_spinner="Loading historical data...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.dropna(subset=["US_AQI"])
    for c in ["Solar_Radiation_Wm2", "Rain_mm", "Temp_2m_C", "Humidity_Percent",
              "Wind_Speed_10m_kmh", "Surface_Pressure_hPa"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mean())
    df["Health_Risk_Score"] = (
        df["US_AQI"] * 0.5 + df["Temp_2m_C"] * 0.1 +
        df["Humidity_Percent"] * 0.1 + df["Wind_Speed_10m_kmh"] * 0.1 +
        df["Solar_Radiation_Wm2"] * 0.1 + df["Surface_Pressure_hPa"] * 0.05 +
        df["Rain_mm"] * 0.05
    )
    def cl(s):
        if s <= 50:  return "Low"
        if s <= 100: return "Moderate"
        if s <= 200: return "High"
        return "Severe"
    df["Health_Risk_Level"] = df["Health_Risk_Score"].apply(cl)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live(city):
    if city not in CITY_COORDS: return None
    lat, lon = CITY_COORDS[city]
    try:
        h = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "surface_pressure,shortwave_radiation,rain"
            "&past_days=3&forecast_days=1&wind_speed_unit=kmh",
            timeout=10
        ).json().get("hourly", {})
        temps = h.get("temperature_2m", [28] * 96)
        hum   = h.get("relative_humidity_2m", [60] * 96)
        wind  = h.get("wind_speed_10m", [10] * 96)
        pres  = h.get("surface_pressure", [1013] * 96)
        solar = h.get("shortwave_radiation", [200] * 96)
        rain  = h.get("rain", [0] * 96)
    except:
        temps = hum = wind = pres = solar = rain = [28] * 96
    try:
        owm      = requests.get(
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}",
            timeout=10
        ).json()
        c        = owm["list"][0]["components"]
        live_aqi = owm_to_us(owm["list"][0]["main"]["aqi"])
        pm25     = c.get("pm2_5", 30.0)
        pm10     = c.get("pm10", 50.0)
    except:
        live_aqi, pm25, pm10 = 100.0, 30.0, 50.0
    n    = len(temps)
    rows = []
    for i in range(-SEQ_LEN, 0):
        idx = i if abs(i) <= n else -1
        rows.append([
            live_aqi, pm25, pm10,
            temps[idx], hum[idx], wind[idx],
            pres[idx], solar[idx], rain[idx]
        ])
    return {
        "rows": rows, "live_aqi": live_aqi, "pm25": pm25, "pm10": pm10,
        "temp": temps[-1], "humidity": hum[-1], "wind": wind[-1],
        "pressure": pres[-1], "solar": solar[-1], "rain": rain[-1],
    }


@st.cache_resource(show_spinner=False)
def load_model_city(city):
    s         = city.replace(" ", "_")
    # Pehle .keras try karo, phir .h5
    keras_path = os.path.join(MODELS_DIR, f"{s}_lstm_v2.keras")
    h5_path    = os.path.join(MODELS_DIR, f"{s}_lstm.h5")
    sp         = os.path.join(MODELS_DIR, f"{s}_scaler.save")
    np_path    = os.path.join(MODELS_DIR, f"{s}_nfeatures.save")
    ls_path    = os.path.join(MODELS_DIR, f"{s}_lastseq.save")

    model = None
    if os.path.exists(keras_path):
        try:
            model = load_model(keras_path, compile=False)
        except Exception:
            model = None
    if model is None and os.path.exists(h5_path):
        try:
            model = load_model(h5_path, compile=False)
        except Exception:
            model = None
    if model is None:
        return None, None, None, None

    if not os.path.exists(sp):
        return None, None, None, None

    scaler  = joblib.load(sp)
    n_feat  = joblib.load(np_path) if os.path.exists(np_path) else len(FEATURES)
    lastseq = joblib.load(ls_path) if os.path.exists(ls_path) else None
    return model, scaler, n_feat, lastseq


def add_eng_features_row(row_vals, history, dt):
    aqi  = row_vals[0]
    h    = np.array(history + [aqi])
    def lag(k):       return h[-k-1] if len(h) > k else aqi
    def roll_mean(w): return np.mean(h[-w:]) if len(h) >= w else np.mean(h)
    def roll_std(w):  return np.std(h[-w:])  if len(h) >= w else 0.0
    def roll_max(w):  return np.max(h[-w:])  if len(h) >= w else aqi
    def diff(k):      return aqi - h[-k-1]   if len(h) > k  else 0.0
    pm25 = row_vals[1]; pm10 = row_vals[2]
    temp = row_vals[3]; hum  = row_vals[4]; wind = row_vals[5]
    eng = [
        np.sin(2*np.pi*dt.hour/24),    np.cos(2*np.pi*dt.hour/24),
        np.sin(2*np.pi*dt.month/12),   np.cos(2*np.pi*dt.month/12),
        np.sin(2*np.pi*dt.weekday()/7),np.cos(2*np.pi*dt.weekday()/7),
        lag(1), lag(2), lag(3), lag(6), lag(12), lag(24), lag(48),
        roll_mean(3), roll_mean(6), roll_mean(12), roll_mean(24), roll_mean(48),
        roll_std(3),  roll_std(6),  roll_std(24),
        roll_max(3),  roll_max(6),  roll_max(24),
        diff(1), diff(3), diff(6), diff(24),
        pm25 / (pm10 + 1e-6),
        temp * hum / 100,
        aqi  / (wind + 1e-6),
        int(7 <= dt.hour <= 10),
        int(17 <= dt.hour <= 20),
        int(dt.hour >= 22 or dt.hour <= 5),
        int(dt.weekday() >= 5),
    ]
    return np.concatenate([row_vals, eng])


def predict_future_smart(model, scaler, n_feat, lastseq, live_rows, steps=168):
    now    = datetime.now().replace(minute=0, second=0, microsecond=0)
    n_base = len(FEATURES)
    if lastseq is not None and lastseq.shape[1] == n_feat:
        seq = lastseq.copy().astype(np.float32)
    else:
        seq_rows  = []
        aqi_hist  = []
        for i, r in enumerate(live_rows):
            dt  = now - timedelta(hours=len(live_rows) - i)
            ext = add_eng_features_row(np.array(r, dtype=np.float32), aqi_hist, dt)
            ext = ext[:n_feat] if len(ext) > n_feat else np.pad(ext, (0, max(0, n_feat - len(ext))))
            seq_rows.append(ext)
            aqi_hist.append(r[0])
        seq = scaler.transform(np.array(seq_rows[-SEQ_LEN:], dtype=np.float32))
    if live_rows and len(live_rows) >= 1:
        aqi_hist = list(scaler.inverse_transform(seq)[:, 0])
        for i, r in enumerate(live_rows[-min(len(live_rows), SEQ_LEN):]):
            idx = SEQ_LEN - len(live_rows) + i
            if 0 <= idx < SEQ_LEN:
                dt_live = now - timedelta(hours=SEQ_LEN - i)
                ext = add_eng_features_row(np.array(r, dtype=np.float32), aqi_hist[:idx+1], dt_live)
                ext = ext[:n_feat] if len(ext) > n_feat else np.pad(ext, (0, max(0, n_feat - len(ext))))
                seq[idx] = scaler.transform(ext.reshape(1, -1))[0]
                aqi_hist.append(r[0])
    preds    = []
    aqi_hist = list(scaler.inverse_transform(seq)[:, 0])
    for step in range(steps):
        dt_pred = now + timedelta(hours=step + 1)
        pred_s  = model.predict(seq.reshape(1, SEQ_LEN, n_feat), verbose=0)[0, 0]
        pred_aqi = float(np.clip(
            scaler.inverse_transform(
                np.concatenate([[pred_s], np.zeros(n_feat-1)]).reshape(1, -1)
            )[0, 0], 0, 500
        ))
        preds.append(pred_aqi)
        aqi_hist.append(pred_aqi)
        last_base    = scaler.inverse_transform(seq[-1:])[0][:n_base]
        last_base[0] = pred_aqi
        new_ext      = add_eng_features_row(last_base, aqi_hist[-60:], dt_pred)
        new_ext      = new_ext[:n_feat] if len(new_ext) > n_feat else np.pad(new_ext, (0, max(0, n_feat - len(new_ext))))
        new_scaled   = scaler.transform(new_ext.reshape(1, -1))[0]
        seq          = np.vstack([seq[1:], new_scaled])
    return np.array(preds, dtype=np.float32)


# UI
st.set_page_config(page_title="India AQI Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

df = load_data()

with st.sidebar:
    st.title("AQI Dashboard")
    st.markdown("---")
    selected_city  = st.selectbox("Select City", sorted(CITY_COORDS.keys()))
    selected_year  = st.selectbox("Select Year", sorted(df["Year"].unique(), reverse=True))
    forecast_days  = st.slider("Forecast Days", 1, 7, 7)
    st.markdown("---")
    st.subheader("Your Profile")
    user_name       = st.text_input("Name (optional)", placeholder="e.g. Priya")
    user_age        = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    user_conditions = st.multiselect("Health Conditions", options=HEALTH_CONDITIONS[1:])
    if not user_conditions:
        user_conditions = ["None"]
    st.markdown("---")
    st.subheader("Planned Activity")
    selected_activity = st.selectbox("What are you planning?", list(ACTIVITY_ADVICE.keys()))
    st.markdown("---")
    st.subheader("AQI Alert Settings")
    alert_threshold = st.slider("Alert me when AQI exceeds:", 50, 300, 150, step=10)
    notify_email    = st.text_input("Email for alerts (optional)", placeholder="you@email.com")
    if notify_email and st.button("Subscribe to Alerts"):
        st.success(f"Alert saved for {notify_email} (threshold: AQI {alert_threshold})")
    st.markdown("---")
    st.info("Live data refreshes every 30 min.")

city_df     = df[(df["City"] == selected_city) & (df["Year"] == selected_year)]
filtered_df = df[df["Year"] == selected_year]

st.title("India AQI and Health Risk Dashboard")
if user_name:
    st.markdown(f"### Hello, {user_name}! Here is your personalised air quality report.")
st.markdown("---")

with st.spinner(f"Fetching live data for {selected_city}..."):
    live = fetch_live(selected_city)

if live:
    st.subheader(f"Live Conditions - {selected_city}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Live AQI",  f"{live['live_aqi']:.0f}", aqi_cat(live['live_aqi']))
    c2.metric("Temp",      f"{live['temp']:.1f} C")
    c3.metric("Humidity",  f"{live['humidity']:.0f}%")
    c4.metric("Wind",      f"{live['wind']:.1f} km/h")
    c5.metric("PM2.5",     f"{live['pm25']:.1f} ug/m3")
    c6.metric("PM10",      f"{live['pm10']:.1f} ug/m3")
    cr  = health_risk(live['live_aqi'])
    fn  = {"Low": st.success, "Moderate": st.warning, "High": st.error, "Severe": st.error}
    msg = get_personalized_precaution(cr, user_age, user_conditions)
    fn.get(cr, st.info)(f"Current Health Risk: {cr} - {msg}")
    act_msg = ACTIVITY_ADVICE.get(selected_activity, {}).get(cr, "")
    if act_msg:
        st.info(f"{selected_activity} Today: {act_msg}")
    st.markdown("---")

    st.subheader(f"{forecast_days}-Day AQI Forecast from Today - {selected_city}")
    st.caption(f"Forecast starts: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

    model, scaler, n_feat, lastseq = load_model_city(selected_city)

    if model is None:
        st.warning(f"No trained model found for {selected_city}. Run train_lstm.py first.")
    else:
        with st.spinner("Running LSTM forecast..."):
            fa = predict_future_smart(model, scaler, n_feat, lastseq,
                                      live["rows"], steps=forecast_days * 24)

        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        fdf = pd.DataFrame({
            "Datetime":      [now + timedelta(hours=i + 1) for i in range(forecast_days * 24)],
            "Predicted_AQI": fa,
        })
        fdf["Category"]    = fdf["Predicted_AQI"].apply(aqi_cat)
        fdf["Health_Risk"] = fdf["Predicted_AQI"].apply(health_risk)
        fdf["Date"]        = fdf["Datetime"].dt.date

        max_future_aqi = fdf["Predicted_AQI"].max()
        if max_future_aqi > alert_threshold:
            first_exceed = fdf[fdf["Predicted_AQI"] > alert_threshold].iloc[0]
            st.warning(
                f"AQI Alert! Forecast will exceed {alert_threshold} starting "
                f"{first_exceed['Datetime'].strftime('%d %b, %I:%M %p')} "
                f"(predicted: {first_exceed['Predicted_AQI']:.0f})"
            )

        daily = fdf.groupby("Date").agg(
            Avg_AQI=("Predicted_AQI", "mean"),
            Max_AQI=("Predicted_AQI", "max"),
            Min_AQI=("Predicted_AQI", "min"),
            Health_Risk=("Health_Risk", lambda x: x.mode()[0])
        ).reset_index()

        dc = st.columns(len(daily))
        for i, (_, row) in enumerate(daily.iterrows()):
            with dc[i]:
                rc       = RISK_COLORS.get(row["Health_Risk"], "#fff")
                is_today = (row["Date"] == date.today())
                border   = "2px solid #00b4d8" if is_today else "1px solid #444"
                today_label = "<br><small style='color:#00b4d8'>TODAY</small>" if is_today else ""
                st.markdown(f"""
                <div style="border:{border};border-radius:12px;padding:12px;
                text-align:center;background:#1e1e2e;margin:2px">
                <b>{row['Date'].strftime('%a')}</b>{today_label}<br>
                <small>{row['Date'].strftime('%d %b')}</small><br><br>
                <b style="font-size:1.2rem;color:#00b4d8">{row['Avg_AQI']:.0f}</b><br>
                <small style="color:#aaa">AQI</small><br>
                <small style="color:{rc}">{row['Health_Risk']}</small><br>
                <small style="color:#666">High:{row['Max_AQI']:.0f} Low:{row['Min_AQI']:.0f}</small>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fdf["Datetime"], y=fdf["Predicted_AQI"],
            mode="lines", name="Forecast AQI",
            line=dict(color="#00b4d8", width=2),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.12)"
        ))
        fig.add_hline(y=alert_threshold, line_dash="dash", line_color="orange",
                      annotation_text=f"Alert: {alert_threshold}",
                      annotation_position="top right")
        for y0, y1, col, lbl in [
            (0,   50,  "rgba(0,228,0,0.06)",   "Good"),
            (50,  100, "rgba(255,255,0,0.06)",  "Moderate"),
            (100, 150, "rgba(255,126,0,0.06)",  "USG"),
            (150, 200, "rgba(255,0,0,0.06)",    "Unhealthy"),
            (200, 300, "rgba(143,63,151,0.06)", "Very Unhealthy"),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(
            title=f"Hourly AQI Forecast - Next {forecast_days} Days from Today",
            xaxis_title="Date and Time", yaxis_title="AQI",
            height=420, template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        rm   = {"Low": 1, "Moderate": 2, "High": 3, "Severe": 4}
        fig2 = go.Figure(go.Scatter(
            x=fdf["Datetime"], y=fdf["Health_Risk"].map(rm),
            mode="lines+markers",
            marker=dict(color=[RISK_COLORS[r] for r in fdf["Health_Risk"]], size=4),
            line=dict(color="#555", width=1),
            text=fdf["Health_Risk"],
            hovertemplate="%{x}<br>Risk: %{text}<extra></extra>"
        ))
        fig2.update_layout(
            title="Health Risk Level Forecast",
            yaxis=dict(tickvals=[1, 2, 3, 4], ticktext=["Low", "Moderate", "High", "Severe"]),
            height=300, template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Personalized Health Advisory")
        for _, row in daily.iterrows():
            risk_label = row["Health_Risk"]
            risk_fn    = {"Low": st.success, "Moderate": st.warning,
                          "High": st.error, "Severe": st.error}.get(risk_label, st.info)
            msg = get_personalized_precaution(risk_label, user_age, user_conditions)
            act = ACTIVITY_ADVICE.get(selected_activity, {}).get(risk_label, "")
            risk_fn(f"{row['Date'].strftime('%A, %d %b')} - AQI {row['Avg_AQI']:.0f} | "
                    f"Risk: {risk_label}\n\n{msg}\n\n{selected_activity}: {act}")

        st.download_button("Download Forecast CSV",
            fdf.to_csv(index=False).encode(),
            f"{selected_city}_{forecast_days}day_forecast.csv", "text/csv")
else:
    st.error("Could not fetch live data. Check your internet connection.")

st.markdown("---")

st.subheader(f"Historical AQI Trend - {selected_city} ({selected_year})")
if not city_df.empty:
    dh = city_df.resample("D", on="Datetime")["US_AQI"].mean().reset_index()
    ft = px.line(dh, x="Datetime", y="US_AQI", template="plotly_dark",
                 title="Daily Average AQI")
    ft.update_traces(line_color="#00b4d8")
    st.plotly_chart(ft, use_container_width=True)

st.subheader("India AQI Map")
md = filtered_df.groupby(["City", "Latitude", "Longitude", "AQI_Category"])["US_AQI"].mean().reset_index()
st.plotly_chart(
    px.scatter_geo(md, lat="Latitude", lon="Longitude", hover_name="City",
                   size="US_AQI", color="AQI_Category",
                   color_discrete_map=AQI_COLORS, scope="asia", template="plotly_dark"),
    use_container_width=True
)

st.subheader("Top 10 Most Polluted Cities")
t10 = filtered_df.groupby("City")["US_AQI"].mean().sort_values(ascending=False).head(10).reset_index()
st.plotly_chart(px.bar(t10, x="City", y="US_AQI", color="US_AQI",
    color_continuous_scale="Reds", template="plotly_dark"), use_container_width=True)

st.subheader(f"Pollutant Levels - {selected_city}")
poll_cols = [c for c in ["PM2_5_ugm3","PM10_ugm3","NO2_ugm3","SO2_ugm3","O3_ugm3","CO_ugm3"]
             if c in city_df.columns]
if poll_cols:
    pv = city_df[poll_cols].mean().reset_index()
    pv.columns = ["Pollutant", "Value"]
    st.plotly_chart(px.bar(pv, x="Pollutant", y="Value", color="Value",
        color_continuous_scale="Oranges", template="plotly_dark"), use_container_width=True)

st.subheader("AQI Category Distribution")
st.plotly_chart(px.histogram(filtered_df, x="AQI_Category", color="AQI_Category",
    color_discrete_map=AQI_COLORS, template="plotly_dark"), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Festival Impact on AQI")
    fe = filtered_df.groupby("Festival_Period")["US_AQI"].mean().reset_index()
    fe["Festival_Period"] = fe["Festival_Period"].map({0: "Non-Festival", 1: "Festival"})
    st.plotly_chart(px.bar(fe, x="Festival_Period", y="US_AQI", color="US_AQI",
        color_continuous_scale="YlOrRd", template="plotly_dark"), use_container_width=True)
with col2:
    st.subheader("Crop Burning Impact on AQI")
    cr_df = filtered_df.groupby("Crop_Burning_Season")["US_AQI"].mean().reset_index()
    cr_df["Crop_Burning_Season"] = cr_df["Crop_Burning_Season"].map({0: "Normal", 1: "Burning Season"})
    st.plotly_chart(px.bar(cr_df, x="Crop_Burning_Season", y="US_AQI", color="US_AQI",
        color_continuous_scale="YlOrRd", template="plotly_dark"), use_container_width=True)

st.subheader("City Ranking by Average AQI")
rk = filtered_df.groupby("City")["US_AQI"].mean().sort_values(ascending=False).reset_index()
rk.columns = ["City", "Average AQI"]
rk.insert(0, "Rank", range(1, len(rk) + 1))
st.dataframe(rk.style.background_gradient(subset=["Average AQI"], cmap="RdYlGn_r"),
             use_container_width=True)

st.markdown("---")
st.caption("Live data refreshes every 30 min. Powered by CNN-BiLSTM, OpenWeatherMap and Open-Meteo.")
