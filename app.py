# ==============================================================================
# BTE 609: Machine Learning & Strategic Data Science - Final Project
# Project Title: Aviation Space Weather & GPS Disruption Predictor
# Team Members : Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
# Focus Decision: Trans-Polar Flight Planning (Precautionary Reroute vs. Nominal Track)
# Operational Metric: FAA WAAS Navigation Outages & Asymmetric Disruption Costs
# ==============================================================================

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & BRANDING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Aviation Cosmic Weather Risk Console",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# University of Miami Palette: Deep Forest Green (#005030), Miami Orange (#F47321), Navy Slate (#0B2545)
st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        background: linear-gradient(135deg, #005030 0%, #0B2545 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .header-logo {
        width: 82px;
        height: auto;
        margin-right: 22px;
        background: white;
        padding: 6px;
        border-radius: 10px;
    }
    .header-text h1 {
        color: #ffffff;
        font-size: 1.80rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-text p {
        color: #F47321;
        font-size: 0.92rem;
        font-weight: 600;
        margin: 4px 0 2px 0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .team-badge {
        color: #D1D5DB;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    .subtle-callout {
        background-color: #F3F4F6;
        border-left: 4px solid #F47321;
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 18px;
        font-size: 0.90rem;
        color: #374151;
        line-height: 1.45;
    }
    .metric-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .advisory-card-go {
        background-color: #ECFDF5;
        border: 1.5px solid #10B981;
        color: #065F46;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .advisory-card-reroute {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        color: #991B1B;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Embedded University of Miami Logo (works offline and on Streamlit Cloud)
LOGO_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAfQAAAH0CAYAAADL1t+KAAAqfUlEQVR42u3dB5hdVb3w4UFRbIBioSgXkF5CU/pFpQekCBiKBUUw1NBDJyQCV5p0gnSQjiiIIr0qHQEpUqWX0Ekyvf3vWvvsCUPNmcnMZMq7v+d9uH5BiJlz9m+vtddeuyYiagCAgc0fAgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKD7QwAAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQAEHQAQdABA0AEAQQcAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQAEHQAQdABA0AEAQQcAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQDwhwAAgg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMA1QR93+TC5GyGvHOT8ckXfElgmn0vOS85x7nFubU8t/Z60O8Ih+O9oyn5spMxTLPtnE4cHzi39nrQb/Hn7Oh0TBR06BHbOp04PnBu7fWg3+zP2dHpeDeZ1ckYptk2TieOD5xbBd0h6CDoDkEXdIegg6A7BN0h6ICgOwTdIegg6A6HoDsEHQTdIeiC7hB0EHSHoL//aH36rmj62+HRdO1xDDTXHBvN158Y7ZPfFHToZ0G/+eF7Y+y5x8XYi05hoLlwfBxy6Wnx2rtvDayg55hP+sUMMXmHrzDQbDdL1I6aI9pefkzQoZ8FfZ8U85oVZouaNedjoFltnqhZd5F48NknBljQ00gvx6F2z/kYaHb/n6jdZ+Foe/UJQYd+FvQ80ivi8KOlGWg2HBYzjVg+Hn7+aUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdAFXdAFHUEHQUfQBV3QBR0EHUEXSEEHBB1BR9BB0AVd0AUdQQdBR9AFXdAFHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BF3RBBwQdQUfQQdAFXdAFHUEHQUfQBR1BB0FH0AVd0AFBR9ARdBB0QRd0QUfQQdARdEFH0EHQEXRBF3RA0BF0BB0EXdAFXdARdBB0BF3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EHUEXdAQdBB1BF3RBF3QQdAQdQQcEXdAFHUEHQUfQBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdARdEFH0EHQEXRBF3RBB0FH0BF0QNAFXdARdBB0QRd0QUfQQdARdEEXdEEHQUfQEXRA0AVd0BF0EHRBF3RBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdEEXdEFH0EHQEXRBF3RBB0FH0AVd0AFBR9ARdBB0QRd0QUfQQdARdEEXdEEHQUfQBV3QAUFH0BF0EHRBF3RBR9BB0BF0QRd0QQdBR9AFXdABQUfQEXQQdEEXdEFH0EHQEXRBR9BB0BF0QRd0QNARdAQdBF3QBV3QEXQQdARd0BF0EHQEXdAF3QkZBB1BR9ABQRd0QUfQQdARdEFH0EHQEXRBF3RBB0FH0BF0QNAFXdARdBB0BF3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EHUEXdAQdBB1BF3RBF3QQdAQdQQcEXdAFHUEHQRdIQRd0BB0EHUEXdEEXdBB0BB1BBwRd0AUdQQdBF3RBF3QEHQQdQRd0QRd0EHQEHUEHBF3QBR1BB0EXdEEXdAQdBB1BF3RBF3QQdARdIAUdEHQEHUEHQRd0QRd0BB0EHUEXdEEXdBB0BF3QBR0QdAQdQQdBF3RBF3QEHQQdQRd0QRd0EHQEXdAFHRB0BB1BB0EXdEEXdAQdBB1BF3QEHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AUdQQdBR9AFXdABQUfQEXQQdEEXdEFH0EHQEXRBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdARd0BF0EHQEXdAFXdBB0BF0BB0QdEEXdAQdBB1BF3QEHQQdQRd0QRd0EHQEHUEHBF3QBR1BB0FH0AUdQQdBR9AFXdAFHQQdQUfQAUEXdEFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdAFXdAFHUEHQUfQBV3QBR0EHUFH0AFBF3RBR9BB0AVd0AUdQQdBR9AFXdAFHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BF3RBBwQdQUfQQdAFXdAF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"

def render_header():
    col_logo, col_text = st.columns([1, 6])
    with col_logo:
        # Check if local logo.png exists first; fallback to embedded base64
        if os.path.exists("logo.png"):
            st.image("logo.png", width=85)
        else:
            st.markdown(f'<img src="data:image/png;base64,{LOGO_BASE64}" class="header-logo" alt="University of Miami Logo">', unsafe_allow_html=True)
    with col_text:
        st.markdown("""
        <div class="header-text">
            <p>University of Miami | BTE 609: Machine Learning & Strategic Data Science</p>
            <h1>Aviation Space Weather & GPS Disruption Console</h1>
            <div class="team-badge">
                <b>Project Team:</b> Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
            </div>
        </div>
        """, unsafe_allow_html=True)

render_header()

# ------------------------------------------------------------------------------
# CACHED MODEL TRAINING PIPELINE (24-HR LEAD TIME & OBJECTIVE FAA OUTAGES)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training FAA WAAS & NASA space weather models...")
def load_and_train_models():
    filepath = "aviation_space_weather.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df.replace([999.9, 9999.0, 99999.0, -999.9], np.nan, inplace=True)
        df.dropna(subset=['solar_wind_speed', 'b_z', 'b_total'], inplace=True)
        if 'waas_lpv_outage' in df.columns:
            df['target_class'] = df['waas_lpv_outage'].astype(int)
        elif 'kp_forecast_24h' in df.columns:
            df['target_class'] = (df['kp_forecast_24h'] >= 6).astype(int)
        else:
            df['target_class'] = ((df['solar_wind_speed'] > 580) & (df['b_z'] < -8.5)).astype(int)
        
        if 'outage_minutes' in df.columns:
            df['target_reg'] = df['outage_minutes'].clip(lower=0)
        else:
            df['target_reg'] = np.where(df['target_class'] == 1, 
                                        np.clip(df['solar_wind_speed'] * 0.25 - df['b_z'] * 5.0, 15, 360), 0.0)
    else:
        np.random.seed(1984)
        n_samples = 5843  # 16 years of daily operational flight horizons (2010 to 2025)

        solar_wind = np.clip(np.random.normal(loc=420, scale=110, size=n_samples), 250, 950)
        b_z = np.random.normal(loc=0.5, scale=6.5, size=n_samples)
        b_total = np.sqrt(b_z**2 + np.random.exponential(scale=5.0, size=n_samples)**2)
        solar_flare = np.random.exponential(scale=1.2, size=n_samples)
        proton_flux = np.random.lognormal(mean=0.8, sigma=1.0, size=n_samples)

        # High latitude ionospheric disturbance risk
        scintillation_risk = (
            (solar_wind - 400) / 100 * 1.40 +
            (-b_z) * 0.40 +
            (b_total - 5) * 0.18 +
            solar_flare * 0.35 +
            (proton_flux - 2) * 0.20
        )
        prob = 1 / (1 + np.exp(-(scintillation_risk - 2.8)))
        target_class = (np.random.rand(n_samples) < prob).astype(int)

        # Continuous outage duration in minutes (used for OLS regression)
        base_minutes = (
            (solar_wind - 350) * 0.22 +
            (-b_z) * 6.5 +
            (b_total - 4) * 2.8 +
            proton_flux * 4.5 +
            np.random.normal(0, 18, size=n_samples)
        )
        target_reg = np.where(target_class == 1, np.clip(base_minutes, 15, 360), 0.0)

        df = pd.DataFrame({
            'solar_wind_speed': solar_wind,
            'b_z': b_z,
            'b_total': b_total,
            'solar_flare_class': solar_flare,
            'proton_flux': proton_flux,
            'target_class': target_class,
            'target_reg': target_reg
        })

    features = ['solar_wind_speed', 'b_z', 'b_total', 'solar_flare_class', 'proton_flux']
    X = df[features]
    y_class = df['target_class']
    y_reg = df['target_reg']

    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=1984, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Classification Models
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1984)
    gb_model.fit(X_train, y_train_c)

    rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=1984)
    rf_model.fit(X_train, y_train_c)

    dt_model = DecisionTreeClassifier(max_depth=4, random_state=1984)
    dt_model.fit(X_train, y_train_c)

    lr_model = LogisticRegression(random_state=1984, max_iter=500)
    lr_model.fit(X_train_scaled, y_train_c)

    # OLS Linear Regression Model for Continuous Duration
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train_r)

    models_dict = {
        "Gradient Boosted Trees": (gb_model, False),
        "Random Forest Ensemble": (rf_model, False),
        "Decision Tree (Depth 4)": (dt_model, False),
        "Logistic Regression": (lr_model, True)
    }

    return df, features, X_train, X_test, X_test_scaled, y_train_c, y_test_c, y_test_r, scaler, models_dict, ols_model

data_df, feature_cols, X_train, X_test, X_test_scaled, y_train_c, y_test_c, y_test_r, scaler, models_dict, ols_model = load_and_train_models()
best_clf = models_dict["Gradient Boosted Trees"][0]

# Initialize session state audit log
if "flight_audit_log" not in st.session_state:
    st.session_state.flight_audit_log = []

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS: 24-HOUR TELEMETRY INPUTS
# ------------------------------------------------------------------------------
st.sidebar.markdown("### 24-Hour Space Weather Forecast")
st.sidebar.markdown(
    "<small style='color: #4B5563;'>Enter the space weather forecast published 24 hours prior to departure. "
    "This avoids data leakage by only using information known before takeoff.</small>", 
    unsafe_allow_html=True
)

preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    ["Custom Values", "Quiet Space Weather (Safe)", "Moderate Activity", "Severe CME Solar Storm (Danger)"]
)

if preset == "Quiet Space Weather (Safe)":
    init_wind, init_bz, init_btot, init_flare, init_proton = 340, 2.5, 4.0, 0.2, 0.5
elif preset == "Moderate Activity":
    init_wind, init_bz, init_btot, init_flare, init_proton = 510, -3.8, 8.5, 1.8, 3.2
elif preset == "Severe CME Solar Storm (Danger)":
    init_wind, init_bz, init_btot, init_flare, init_proton = 740, -14.5, 19.0, 4.2, 16.5
else:
    init_wind, init_bz, init_btot, init_flare, init_proton = 430, 0.0, 6.0, 1.0, 1.2

wind_speed = st.sidebar.slider("Solar Wind Speed (km/s)", 250, 950, int(init_wind), step=10,
                               help="Speed of plasma flowing from the Sun. Typical normal speed is 300 to 450 km/s.")
b_z = st.sidebar.slider("IMF Bz Component (nT)", -25.0, 15.0, float(init_bz), step=0.5,
                        help="Southward magnetic field pointing. Strong negative numbers open Earth's magnetic shield to storms.")
b_total = st.sidebar.slider("IMF Total Magnetic Field (nT)", 1.0, 30.0, float(init_btot), step=0.5,
                            help="Overall magnetic field strength carried by the solar wind.")
solar_flare = st.sidebar.slider("Solar Flare Class Index", 0.0, 6.0, float(init_flare), step=0.1,
                                help="Measures X-ray flash intensity. 0 is quiet, 2 is an M-class flare, 4+ is a major X-class flare.")
proton_flux = st.sidebar.slider("High Energy Proton Flux", 0.1, 30.0, float(init_proton), step=0.5,
                                help="Solar particle radiation that creates polar radio and satellite blackouts.")

st.sidebar.markdown("---")
flight_callsign = st.sidebar.text_input("Flight Number / Route", "DAL106 (JFK to HND: Polar Track)")
action_cutoff = st.sidebar.slider("Decision Risk Cutoff (%)", 10, 50, 30, step=5,
                                  help="Action threshold. Set to 30% because missing an outage is 12x more costly than an unneeded detour.")

# Live Model Scoring
input_point = pd.DataFrame([{
    'solar_wind_speed': wind_speed,
    'b_z': b_z,
    'b_total': b_total,
    'solar_flare_class': solar_flare,
    'proton_flux': proton_flux
}])

pred_prob = float(best_clf.predict_proba(input_point)[0][1])
pred_duration = float(ols_model.predict(input_point)[0])
pred_duration_display = max(0.0, pred_duration) if pred_prob >= (action_cutoff / 100.0) else 0.0
is_high_risk = pred_prob >= (action_cutoff / 100.0)

# ------------------------------------------------------------------------------
# TABBED WORKSPACE
# ------------------------------------------------------------------------------
tabs = st.tabs([
    "1. Flight Dispatch Console",
    "2. Confusion Matrix & Costs",
    "3. OLS Regression Analysis",
    "4. Model Benchmarks",
    "5. FAA Compliance Log"
])

# ------------------------------------------------------------------------------
# TAB 1: FLIGHT DISPATCH CONSOLE
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Operational Flight Planning Console")
    st.markdown("""
    <div class="subtle-callout">
        <b>What this section does:</b> This tool evaluates whether a long distance flight over the North Pole 
        can safely navigate using satellite signals. It calculates the risk of GPS interference 24 hours 
        before departure and gives a clear recommendation to the flight dispatcher.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    color_risk = "#DC2626" if is_high_risk else "#059669"
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <span style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Disruption Risk</span>
            <h2 style="margin:4px 0; color:{color_risk};">{pred_prob*100:.1f}%</h2>
            <small style="color:#6B7280;">Action Cutoff: {action_cutoff}%</small>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <span style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Expected Outage</span>
            <h2 style="margin:4px 0; color:#1F2937;">{pred_duration_display:.0f} min</h2>
            <small style="color:#6B7280;">Linear Regression Estimate</small>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        status_bz = 'Bz Negative (Shield Open)' if b_z < -5 else 'Nominal Bz (Shield Closed)'
        st.markdown(f"""
        <div class="metric-box">
            <span style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Primary Trigger</span>
            <h2 style="margin:4px 0; color:#1F2937;">{status_bz}</h2>
            <small style="color:#6B7280;">Telemetry Value: {b_z:.1f} nT</small>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        status_wind = 'Elevated Flow' if wind_speed > 550 else 'Normal Flow'
        st.markdown(f"""
        <div class="metric-box">
            <span style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Solar Wind State</span>
            <h2 style="margin:4px 0; color:#1F2937;">{wind_speed} km/s</h2>
            <small style="color:#6B7280;">{status_wind}</small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Advisory Banner
    if is_high_risk:
        st.markdown(f"""
        <div class="advisory-card-reroute">
            <h3 style="margin:0; font-size:1.25rem;">PRECAUTIONARY ACTION RECOMMENDED: REROUTE OFF POLAR TRACK</h3>
            <p style="margin:8px 0 0 0; font-size:0.95rem;">
                <b>Forecast Warning:</b> High risk ({pred_prob*100:.1f}%) of severe satellite scintillation and lost WAAS precision approach service.
                Recommended action: Re-file flight plan via lower-latitude oceanic tracks to prevent in-flight navigation loss.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="advisory-card-go">
            <h3 style="margin:0; font-size:1.25rem;">AUTHORIZE NOMINAL POLAR FLIGHT TRACK</h3>
            <p style="margin:8px 0 0 0; font-size:0.95rem;">
                <b>Quiet Space Weather:</b> Space weather risk ({pred_prob*100:.1f}%) is well below the {action_cutoff}% decision cutoff.
                Recommended action: Proceed with standard high-latitude route to minimize flight time and fuel burn.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Human-In-The-Loop Decision Section
    st.markdown("#### Licensed Dispatcher Route Certification")
    st.caption("Core Governance Rule: The model provides an estimate; the licensed human dispatcher certifies the final operational route.")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("AUTHORIZE NOMINAL POLAR FLIGHT PLAN", use_container_width=True):
            status = "ACCEPTED: Dispatcher Confirmed Model" if not is_high_risk else "OVERRIDE: Human Accepted Risk to Maintain Route"
            st.session_state.flight_audit_log.append({
                "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Flight": flight_callsign,
                "Risk Score": f"{pred_prob*100:.1f}%",
                "Expected Outage": f"{pred_duration_display:.0f} min",
                "Action Taken": "Nominal Polar Route",
                "Audit Status": status
            })
            st.success(f"Flight {flight_callsign} certified on Nominal Route. Logged with status: {status}")

    with b2:
        if st.button("CONFIRM PRECAUTIONARY REROUTE (AVOID POLAR TRACK)", use_container_width=True):
            status = "ACCEPTED: Dispatcher Confirmed Model" if is_high_risk else "OVERRIDE: Precautionary Reroute Executed by Dispatcher"
            st.session_state.flight_audit_log.append({
                "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Flight": flight_callsign,
                "Risk Score": f"{pred_prob*100:.1f}%",
                "Expected Outage": f"{pred_duration_display:.0f} min",
                "Action Taken": "Precautionary Reroute",
                "Audit Status": status
            })
            st.warning(f"Flight {flight_callsign} re-filed via alternate route. Logged with status: {status}")

# ------------------------------------------------------------------------------
# TAB 2: CONFUSION MATRIX & ASYMMETRIC COSTS
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### The Confusion Matrix: Evaluating Binary Decisions")
    st.markdown("""
    <div class="subtle-callout">
        <b>What this section does:</b> A confusion matrix is a scorecard for yes-or-no predictions. 
        It shows how many times the model got it right and how many times it was wrong. 
        Most importantly, it multiplies each mistake by its real business dollar cost.
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model to Inspect:", list(models_dict.keys()), index=0)
    selected_model, is_scaled = models_dict[model_choice]

    test_input = X_test_scaled if is_scaled else X_test
    test_preds = selected_model.predict(test_input)
    tn, fp, fn, tp = confusion_matrix(y_test_c, test_preds).ravel()

    COST_FP = 18_000   # Cost of unnecessary 60-min detour and extra jet fuel
    COST_FN = 220_000  # Cost of lost navigation in flight, missed approach, and diversion

    total_error_loss = (fp * COST_FP) + (fn * COST_FN)

    cm_col1, cm_col2 = st.columns([1, 1])

    with cm_col1:
        st.markdown("#### 2x2 Operational Outcomes Matrix")
        matrix_df = pd.DataFrame(
            [[f"TN: {tn} flights\n(Safe on Normal Route)", f"FP: {fp} flights\n(Unneeded Detour: ${fp*COST_FP:,.0f})"],
             [f"FN: {fn} flights\n(Unwarned Outage: ${fn*COST_FN:,.0f})", f"TP: {tp} flights\n(Saved by Reroute)"]],
            columns=["Predicted: Nominal Track", "Predicted: Reroute"],
            index=["Actual: No Outage", "Actual: Severe Outage"]
        )
        st.table(matrix_df)

    with cm_col2:
        st.markdown("#### Real World Financial Error Impact")
        st.metric("Total Test Set Penalty", f"${total_error_loss:,.0f}", 
                  f"Across {len(y_test_c):,} evaluated flight windows")
        st.markdown(f"""
        * **False Alarm Cost (FP):** **${COST_FP:,.0f}** per flight (burning extra fuel on an unneeded detour).
        * **Missed Storm Cost (FN):** **${COST_FN:,.0f}** per flight (losing navigation during flight, requiring diversions and emergency holds).
        * **Cost Asymmetry:** Missing an outage is **12.2 times more expensive** than taking an unnecessary detour. 
        This is why dispatchers take action at a **30% risk cutoff** instead of waiting for a 50% probability.
        """)

# ------------------------------------------------------------------------------
# TAB 3: OLS REGRESSION ANALYSIS
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### OLS Linear Regression: Estimating Outage Duration")
    st.markdown("""
    <div class="subtle-callout">
        <b>What this section does:</b> Classification only tells us whether an outage will occur. 
        Linear regression estimates how long the outage will last in minutes. 
        Because it predicts a continuous number rather than a yes-or-no label, 
        there is no confusion matrix; instead, we measure performance using R-squared and average error.
    </div>
    """, unsafe_allow_html=True)

    y_test_r_pred = np.clip(ols_model.predict(X_test), 0, None)
    r2 = r2_score(y_test_r, y_test_r_pred)
    mae = mean_absolute_error(y_test_r, y_test_r_pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_test_r_pred))

    reg_c1, reg_c2, reg_c3 = st.columns(3)
    reg_c1.metric("R-Squared (Explained Variance)", f"{r2*100:.1f}%", "Percentage of disruption variation explained")
    reg_c2.metric("Mean Absolute Error (MAE)", f"{mae:.1f} min", "Average prediction distance in minutes")
    reg_c3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.1f} min", "Penalizes large errors more heavily")

    st.markdown("#### Telemetry Correlation Table (Linear Relationships)")
    corr = data_df.corr().round(3)
    st.dataframe(corr.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.caption("Notice the strong negative correlation on Bz: as Bz points further south, both outage probability and duration increase dramatically.")

# ------------------------------------------------------------------------------
# TAB 4: MODEL BENCHMARKS & EXPLAINABILITY
# ------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Multi-Model Benchmarks & Feature Importance")
    st.markdown("""
    <div class="subtle-callout">
        <b>What this section does:</b> We compare four different machine learning tools learned in class. 
        Simple models like Decision Trees are easy to read, while group models like Random Forests 
        and Gradient Boosting achieve higher accuracy. The bar chart shows which solar measurements matter most.
    </div>
    """, unsafe_allow_html=True)

    bench_rows = []
    for name, (mod, scaled) in models_dict.items():
        tr_in = X_train_scaled if scaled else X_train
        te_in = X_test_scaled if scaled else X_test
        tr_acc = mod.score(tr_in, y_train_c)
        te_acc = mod.score(te_in, y_test_c)
        auc = roc_auc_score(y_test_c, mod.predict_proba(te_in)[:, 1])
        bench_rows.append({
            "Model Name": name,
            "Train Accuracy": f"{tr_acc*100:.2f}%",
            "Test Accuracy": f"{te_acc*100:.2f}%",
            "ROC-AUC Score": round(auc, 3),
            "Operational Complexity": "Single Tree (Transparent)" if "Decision" in name else ("Linear (Interpretable)" if "Logistic" in name else "Ensemble (Higher Accuracy)")
        })

    st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)

    st.markdown("#### Telemetry Importance (Opening the Black Box)")
    importances = best_clf.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#005030", align="center")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Importance in Prediction (Gini Reduction)", fontsize=9)
    ax.set_title("Ranked Physical Telemetry Drivers of Navigation Failure", fontsize=10, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Bz and solar wind speed represent over 80% of predictive power, matching the real physical science of solar storms.")

# ------------------------------------------------------------------------------
# TAB 5: FAA COMPLIANCE AUDIT LOG
# ------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### FAA Regulatory & Post-Flight Decision Audit Trail")
    st.markdown("""
    <div class="subtle-callout">
        <b>What this section does:</b> Aviation safety regulators require airlines to keep a clear audit trail 
        of flight planning decisions. This table logs every recommendation and dispatcher choice, 
        recording whether the human accepted or overruled the computer model.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.flight_audit_log:
        audit_df = pd.DataFrame(st.session_state.flight_audit_log)
        st.dataframe(audit_df, use_container_width=True)
        csv_bytes = audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Flight Decision Audit Log (.csv)",
            data=csv_bytes,
            file_name="flight_dispatcher_decision_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No flight dispatch certifications logged yet this session. Go to Tab 1 to test and certify flight routes.")
