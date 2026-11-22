# ==============================================================================
# BTE 609: Artificial Intelligence for Business
# Professor: Dr. Niam Yaraghi
# Team: Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
# Focus: Aviation Space Weather & Polar Flight Disruption Decision Support
# ==============================================================================

import os
from datetime import datetime, timezone
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------------------
# 1. PAGE SETUP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Cosmic Flight Dispatch | BTE 609",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embedded Base64 logo for guaranteed rendering on Streamlit Cloud
EMBEDDED_LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAfQAAAH0CAYAAADL1t+KAAAqfUlEQVR42u3dB5hdVb3w4UFRbIBioSgXkF5CU/pFpQekCBiKBUUw1NBDJyQCV5p0gnSQjiiIIr0qHQEpUqWX0Ekyvf3vWvvsCUPNmcnMZMq7v+d9uH5BiJlz9m+vtddeuyYiagCAgc0fAgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKD7QwAAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQAEHQAQdABA0AEAQQcAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQAEHQAQdABA0AEAQQcAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQDwhwAAgg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMA1QR93+TC5GyGvHOT8ckXfElgmn0vOS85x7nFubU8t/Z60O8Ih+O9oyn5spMxTLPtnE4cHzi39nrQb/Hn7Oh0TBR06BHbOp04PnBu7fWg3+zP2dHpeDeZ1ckYptk2TieOD5xbBd0h6CDoDkEXdIegg6A7BN0h6ICgOwTdIegg6A6HoDsEHQTdIeiC7hB0EHSHoL//aH36rmj62+HRdO1xDDTXHBvN158Y7ZPfFHToZ0G/+eF7Y+y5x8XYi05hoLlwfBxy6Wnx2rtvDayg55hP+sUMMXmHrzDQbDdL1I6aI9pefkzQoZ8FfZ8U85oVZouaNedjoFltnqhZd5F48NknBljQ00gvx6F2z/kYaHb/n6jdZ+Foe/UJQYd+FvQ80ivi8KOlGWg2HBYzjVg+Hn7+aUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdAFXdAFHUEHQUfQBV3QBR0EHUEXSEEHBB1BR9BB0AVd0AUdQQdBR9AFXdAFHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BF3RBBwQdQUfQQdAFXdAFHUEHQUfQBR1BB0FH0AVd0AFBR9ARdBB0QRd0QUfQQdARdEFH0EHQEXRBF3RA0BF0BB0EXdAFXdARdBB0BF3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EHUEXdAQdBB1BF3RBF3QQdAQdQQcEXdAFHUEHQUfQBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdARdEFH0EHQEXRBF3RBB0FH0BF0QNAFXdARdBB0QRd0QUfQQdARdEEXdEEHQUfQEXRA0AVd0BF0EHRBF3RBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdEEXdEFH0EHQEXRBF3RBB0FH0AVd0AFBR9ARdBB0QRd0QUfQQdARdEEXdEEHQUfQBV3QAUFH0BF0EHRBF3RBR9BB0BF0QRd0QQdBR9AFXdABQUfQEXQQdEEXdEFH0EHQEXRBR9BB0BF0QRd0QNARdAQdBF3QBV3QEXQQdARd0BF0EHQEXdAF3QkZBB1BR9ABQRd0QUfQQdARdEFH0EHQEXRBF3RBB0FH0BF0QNAFXdARdBB0BF3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EHUEXdAQdBB1BF3RBF3QQdAQdQQcEXdAFHUEHQRdIQRd0BB0EHUEXdEEXdBB0BB1BBwRd0AUdQQdBF3RBF3QEHQQdQRd0QRd0EHQEHUEHBF3QBR1BB0EXdEEXdAQdBB1BF3RBF3QQdARdIAUdEHQEHUEHQRd0QRd0BB0EHUEXdEEXdBB0BF3QBR0QdAQdQQdBF3RBF3QEHQQdQRd0QRd0EHQEXdAFHRB0BB1BB0EXdEEXdAQdBB1BF3QEHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AUdQQdBR9AFXdABQUfQEXQQdEEXdEFH0EHQEXRBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdARd0BF0EHQEXdAFXdBB0BF0BB0QdEEXdAQdBB1BF3QEHQQdQRd0QRd0EHQEHUEHBF3QBR1BB0FH0AUdQQdBR9AFXdAFHQQdQUfQAUEXdEFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdAFXdAFHUEHQUfQBV3QBR0EHUFH0AFBF3RBR9BB0AVd0AUdQQdBR9AFXdAFHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BF3RBBwQdQUfQQdAFXdAF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"

def get_logo_html():
    if os.path.exists("logo.png"):
        try:
            with open("logo.png", "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f'<img src="data:image/png;base64,{encoded}" style="height:62px; width:auto; vertical-align:middle; margin-right:16px;">'
        except Exception:
            pass
    return f'<img src="data:image/png;base64,{EMBEDDED_LOGO_B64}" style="height:62px; width:auto; vertical-align:middle; margin-right:16px;">'

# ------------------------------------------------------------------------------
# 2. STYLED CSS (CLEAN, PROFESSIONAL, NO HARSH BORDERS OR WEIRD PANEL TINTS)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean layout and typography */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Executive Header */
    .header-box {
        display: flex;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .header-text h1 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }
    .header-text .subtitle {
        font-size: 0.88rem;
        color: #475569;
        margin: 0;
        line-height: 1.4;
    }
    .team-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #1e293b;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        margin-top: 6px;
    }
    
    /* Clean Cards */
    .clean-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .card-desc {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    /* Status Banners */
    .banner-nominal {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 14px 18px;
        color: #166534;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 12px 0;
    }
    .banner-reroute {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 14px 18px;
        color: #991b1b;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 12px 0;
    }
    
    /* HITL Adaptive Learning Callout */
    .hitl-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 14px 0;
    }
    .hitl-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0369a1;
        margin-bottom: 4px;
    }
    .hitl-text {
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.45;
        margin: 0;
    }

    /* Metric Display */
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. HEADER RENDERING
# ------------------------------------------------------------------------------
logo_tag = get_logo_html()
st.markdown(f"""
<div class="header-box">
    {logo_tag}
    <div class="header-text">
        <h1>Polar Flight Space Weather Risk Console</h1>
        <div class="subtitle">
            <strong>BTE 609: Artificial Intelligence for Business</strong> &bull; Professor: <strong>Dr. Niam Yaraghi</strong>
        </div>
        <div class="team-badge">
            Team: Alex Fotiadis &bull; Dainiuris "Nuri" Oler &bull; Justin Perez &bull; George Kurtzer
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 4. DATA ENGINE & CACHED MACHINE LEARNING PIPELINE
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training predictive models on space weather records...")
def train_aviation_system():
    np.random.seed(1984)
    n_days = 5843  # 16 years of daily operational windows (2010 to 2025)

    solar_wind_speed = np.clip(np.random.normal(loc=420, scale=110, size=n_days), 250, 950)
    b_z = np.random.normal(loc=0.5, scale=6.5, size=n_days)
    b_total = np.sqrt(b_z**2 + np.random.exponential(scale=5.0, size=n_days)**2)
    solar_flare_class = np.random.exponential(scale=1.2, size=n_days)
    proton_flux = np.random.lognormal(mean=0.8, sigma=1.0, size=n_days)

    ionospheric_risk = (
        (solar_wind_speed - 400) / 100 * 1.4 +
        (-b_z) * 0.40 +
        (b_total - 5) * 0.18 +
        solar_flare_class * 0.35 +
        (proton_flux - 2) * 0.20
    )
    prob = 1 / (1 + np.exp(-(ionospheric_risk - 2.8)))
    target_class = (np.random.rand(n_days) < prob).astype(int)

    base_minutes = np.maximum(0, (ionospheric_risk - 1.2) * 45)
    noise_minutes = np.random.normal(0, 18, size=n_days)
    target_duration = np.where(target_class == 1, np.clip(base_minutes + noise_minutes, 15, 360), 0.0)

    df = pd.DataFrame({
        'solar_wind_speed': solar_wind_speed,
        'b_z': b_z,
        'b_total': b_total,
        'solar_flare_class': solar_flare_class,
        'proton_flux': proton_flux,
        'gps_disruption_target': target_class,
        'outage_duration_minutes': target_duration
    })

    features = ['solar_wind_speed', 'b_z', 'b_total', 'solar_flare_class', 'proton_flux']
    X = df[features]
    y_class = df['gps_disruption_target']
    y_reg = df['outage_duration_minutes']

    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=1984, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Classifiers
    dt = DecisionTreeClassifier(max_depth=4, random_state=1984)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=1984)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1984)
    gb.fit(X_train, y_train)

    lr = LogisticRegression(random_state=1984)
    lr.fit(X_train_scaled, y_train)

    # Regressor
    reg = LinearRegression()
    reg.fit(X_train, y_reg_train)

    return {
        'df': df,
        'features': features,
        'X_test': X_test,
        'y_test': y_test,
        'y_reg_test': y_reg_test,
        'scaler': scaler,
        'dt': dt,
        'rf': rf,
        'gb': gb,
        'lr': lr,
        'reg': reg
    }

pipeline = train_aviation_system()
gb_model = pipeline['gb']
reg_model = pipeline['reg']

# ------------------------------------------------------------------------------
# 5. SESSION STATE FOR HUMAN IN THE LOOP (HITL) AND ACTIVE LEARNING
# ------------------------------------------------------------------------------
if "decision_log" not in st.session_state:
    st.session_state.decision_log = []

if "adaptive_threshold" not in st.session_state:
    st.session_state.adaptive_threshold = 0.30

# Recalculate adaptive threshold based on human overrides
overrides = [d for d in st.session_state.decision_log if d["audit_status"] == "OVERRIDE"]
accepts = [d for d in st.session_state.decision_log if d["audit_status"] == "ACCEPTED"]

# If human dispatchers consistently override model warnings on borderline conditions,
# the system nudges the threshold upward to avoid unnecessary reroutes and alert fatigue.
if len(st.session_state.decision_log) >= 2:
    override_nominal_count = sum(1 for d in overrides if d["action"] == "NOMINAL TRACK")
    override_reroute_count = sum(1 for d in overrides if d["action"] == "PRECAUTIONARY REROUTE")
    adjustment = (override_nominal_count * 0.02) - (override_reroute_count * 0.02)
    st.session_state.adaptive_threshold = float(np.clip(0.30 + adjustment, 0.20, 0.45))

active_threshold = st.session_state.adaptive_threshold

# ------------------------------------------------------------------------------
# 6. SIDEBAR CONTROLS (NO CLUTTER, PRACTICAL TELEMETRY SLIDERS)
# ------------------------------------------------------------------------------
st.sidebar.markdown("### Telemetry Inputs (24h Lead)")
st.sidebar.caption("Measurements available 24 hours prior to departure.")

preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    ["Custom Inputs", "Quiet Sun (Nominal)", "Elevated CME Activity", "Severe Storm (G4/G5)"]
)

if preset == "Quiet Sun (Nominal)":
    default_wind, default_bz, default_btot, default_flare, default_proton = 340, 2.5, 4.0, 0.2, 0.6
elif preset == "Elevated CME Activity":
    default_wind, default_bz, default_btot, default_flare, default_proton = 550, -6.8, 11.0, 1.8, 4.5
elif preset == "Severe Storm (G4/G5)":
    default_wind, default_bz, default_btot, default_flare, default_proton = 760, -15.5, 20.0, 4.2, 18.0
else:
    default_wind, default_bz, default_btot, default_flare, default_proton = 430, -1.2, 6.5, 0.8, 1.5

input_wind = st.sidebar.slider("Solar Wind Speed (km/s)", 250, 950, int(default_wind), 10)
input_bz = st.sidebar.slider("Interplanetary Magnetic Field Bz (nT)", -25.0, 25.0, float(default_bz), 0.5)
input_btot = st.sidebar.slider("Total Magnetic Field Btot (nT)", 1.0, 35.0, float(default_btot), 0.5)
input_flare = st.sidebar.slider("Solar Flare Class (GOES Index)", 0.0, 6.0, float(default_flare), 0.1)
input_proton = st.sidebar.slider("Proton Flux (MeV Index)", 0.1, 30.0, float(default_proton), 0.5)

flight_number = st.sidebar.text_input("Flight Number", value="UAL895 (ORD to HKG)")

# Predict with primary Gradient Boosted model
input_df = pd.DataFrame([{
    'solar_wind_speed': input_wind,
    'b_z': input_bz,
    'b_total': input_btot,
    'solar_flare_class': input_flare,
    'proton_flux': input_proton
}])

pred_prob = float(gb_model.predict_proba(input_df)[0][1])
pred_mins = float(np.clip(reg_model.predict(input_df)[0], 0, 360))
rec_reroute = pred_prob >= active_threshold

# ------------------------------------------------------------------------------
# 7. MAIN TABS INTERFACE
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Flight Dispatch Console",
    "2. Why Polar Routes?",
    "3. Confusion Matrix & Costs",
    "4. Outage Duration Regression",
    "5. Model Benchmarks & Audit"
])

# ------------------------------------------------------------------------------
# TAB 1: FLIGHT DISPATCH CONSOLE
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Real-Time Polar Flight Assessment</div>
        <div class="card-desc">
            This console evaluates incoming 24-hour space weather forecasts to determine if a polar flight path 
            is at risk of losing satellite navigation. The system provides an advisory, while a licensed dispatcher 
            certifies the final operational route.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num">{pred_prob*100:.1f}%</div>
            <div class="metric-label">Disruption Probability</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num">{pred_mins:.0f} min</div>
            <div class="metric-label">Estimated Outage Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num">{active_threshold*100:.0f}%</div>
            <div class="metric-label">Active Risk Cutoff</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        decision_label = "Reroute Advised" if rec_reroute else "Polar Route Clear"
        status_color = "#991b1b" if rec_reroute else "#166534"
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num" style="color: {status_color}; font-size:1.35rem; padding-top:4px;">{decision_label}</div>
            <div class="metric-label">System Recommendation</div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation Banner
    if rec_reroute:
        st.markdown(f"""
        <div class="banner-reroute">
            Advisory: Precautionary Polar Reroute Advised. Severe ionospheric scintillation expected. Estimated GPS approach loss of {pred_mins:.0f} minutes.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="banner-nominal">
            Advisory: Authorize Nominal Polar Track. Space weather conditions remain within safe navigation tolerance limits.
        </div>
        """, unsafe_allow_html=True)

    # Human-in-the-Loop Certification Section
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Human in the Loop: Route Certification</div>
        <div class="card-desc">
            Prediction is not a decision. The algorithm estimates probability; the licensed flight dispatcher 
            certifies the legal flight plan. Below, record your operational decision. The model logs your choice 
            and actively refines its recommendations based on human experience.
        </div>
    </div>
    """, unsafe_allow_html=True)

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("Authorize Nominal Polar Track", use_container_width=True):
            status = "ACCEPTED" if not rec_reroute else "OVERRIDE"
            st.session_state.decision_log.insert(0, {
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "flight": flight_number,
                "wind": input_wind,
                "bz": input_bz,
                "probability": f"{pred_prob*100:.1f}%",
                "action": "NOMINAL TRACK",
                "audit_status": status
            })
            st.rerun()

    with bcol2:
        if st.button("Order Precautionary Reroute", use_container_width=True):
            status = "ACCEPTED" if rec_reroute else "OVERRIDE"
            st.session_state.decision_log.insert(0, {
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "flight": flight_number,
                "wind": input_wind,
                "bz": input_bz,
                "probability": f"{pred_prob*100:.1f}%",
                "action": "PRECAUTIONARY REROUTE",
                "audit_status": status
            })
            st.rerun()

    # Active Learning & HITL Feedback Status
    st.markdown(f"""
    <div class="hitl-box">
        <div class="hitl-title">Human-in-the-Loop Active Learning Feedback</div>
        <p class="hitl-text">
            Total human certifications logged: <strong>{len(st.session_state.decision_log)}</strong> | 
            Current adaptive cutoff: <strong>{active_threshold*100:.1f}%</strong>.<br>
            When dispatchers override conservative recommendations during borderline telemetry days, 
            the system learns from human operational experience and gently tunes its cutoff to prevent alert fatigue.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: WHY POLAR ROUTES?
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">The Science and Economics of Polar Flying</div>
        <div class="card-desc">
            A plain language explanation of why solar storms target polar flight corridors and how airlines manage the risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("""
        <div class="clean-card">
            <div class="card-title">1. Earth's Magnetic Shield Has Open Funnels at the Poles</div>
            <div class="card-desc">
                Earth is protected by a magnetic field that acts like an invisible bubble. Near the equator and mid-latitudes, 
                this magnetic field lines up parallel to the surface, smoothly deflecting harmful solar radiation away into space.<br><br>
                However, at the North and South Poles, the magnetic field lines curve straight down into the atmosphere like giant funnels. 
                When a solar storm hits, billions of high-energy charged particles slide directly down these funnels into polar airspace, 
                creating strong electrical disturbances in the upper atmosphere.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clean-card">
            <div class="card-title">2. Satellite Signals and Radio Blackouts</div>
            <div class="card-desc">
                When solar particles pour into the polar air, they create ripples and turbulence in the ionosphere, a process known as 
                ionospheric scintillation.<br><br>
                This turbulence bends and scatters GPS satellite signals passing through to aircraft below, causing navigation errors or 
                disabling precision approach tools (such as FAA WAAS). At the same time, high-frequency (HF) radio signals are absorbed, 
                leaving pilots with limited long-range communications over the Arctic.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with pcol2:
        st.markdown("""
        <div class="clean-card">
            <div class="card-title">3. Why Do Airlines Fly Over the North Pole?</div>
            <div class="card-desc">
                Because Earth is a sphere, the shortest route between North America and East Asia (for example, Chicago to Hong Kong or 
                New York to Tokyo) curves directly across the Arctic Ocean. These are known as Great Circle routes.<br><br>
                Flying over the North Pole saves airlines up to <strong>two full hours of flight time</strong> and thousands of gallons of 
                expensive jet fuel per trip. It is one of the most efficient international corridors in modern commercial aviation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clean-card">
            <div class="card-title">4. The Dispatcher's Economic Dilemma</div>
            <div class="card-desc">
                When space weather telemetry warns of incoming solar storm activity, dispatchers face a critical operational choice:<br><br>
                &bull; <strong>Plan a Precautionary Detour ($18,000 cost):</strong> Route the plane south of the Arctic along lower latitudes. 
                This burns extra fuel and adds an hour of flight time, but completely bypasses the blackout zone.<br>
                &bull; <strong>Fly the Polar Route and Risk a Blackout ($220,000 cost):</strong> If a storm hits mid-flight, the plane loses precision 
                landing guidance and long-range communications, forcing emergency mid-air diversions and holding patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: CONFUSION MATRIX & ASYMMETRIC COSTS
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Classification Performance and Business Impact</div>
        <div class="card-desc">
            A confusion matrix tracks correct predictions against errors. In aviation, errors are highly asymmetric: 
            missing a storm is 12 times more expensive than taking an unnecessary detour.
        </div>
    </div>
    """, unsafe_allow_html=True)

    preds_gb = gb_model.predict(pipeline['X_test'])
    tn, fp, fn, tp = confusion_matrix(pipeline['y_test'], preds_gb).ravel()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num" style="color:#166534;">{tp}</div>
            <div class="metric-label">True Positives (Reroutes)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Protected from severe outages</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num" style="color:#166534;">{tn}</div>
            <div class="metric-label">True Negatives (Nominal)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Maintained fast polar track</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num" style="color:#b45309;">{fp}</div>
            <div class="metric-label">False Positives (Detours)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">$18,000 unneeded fuel burn each</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num" style="color:#991b1b;">{fn}</div>
            <div class="metric-label">False Negatives (Missed)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">$220,000 diversion cost each</div>
        </div>
        """, unsafe_allow_html=True)

    # Cost Matrix Table
    fp_cost = fp * 18000
    fn_cost = fn * 220000
    total_penalty = fp_cost + fn_cost

    st.markdown(f"""
    <div class="clean-card">
        <div class="card-title">Asymmetric Error Cost Breakdown (1,169 Withheld Flight Days)</div>
        <table style="width:100%; border-collapse: collapse; font-size: 0.88rem;">
            <tr style="border-bottom: 1px solid #e2e8f0; text-align: left; color:#64748b;">
                <th style="padding: 8px 0;">Error Category</th>
                <th style="padding: 8px 0;">Count</th>
                <th style="padding: 8px 0;">Unit Cost</th>
                <th style="padding: 8px 0;">Total Financial Impact</th>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 10px 0;"><strong>False Alarm (Unneeded Polar Detour)</strong></td>
                <td>{fp} flights</td>
                <td>$18,000</td>
                <td><strong>${fp_cost:,.0f}</strong></td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 10px 0; color:#991b1b;"><strong>Missed Storm (In-Flight Navigation Loss)</strong></td>
                <td>{fn} flights</td>
                <td>$220,000</td>
                <td style="color:#991b1b;"><strong>${fn_cost:,.0f}</strong></td>
            </tr>
            <tr style="font-size: 0.95rem;">
                <td style="padding: 12px 0;"><strong>Net Operational Disruption Cost</strong></td>
                <td>--</td>
                <td>--</td>
                <td><strong>${total_penalty:,.0f}</strong></td>
            </tr>
        </table>
        <div class="card-desc" style="margin-top:12px;">
            Because a missed outage is 12.2 times more expensive than an unnecessary detour, the model does not wait for a 50% 
            probability to advise action. It uses an operational threshold near 30% to shield the airline from costly diversions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: OUTAGE DURATION REGRESSION
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">OLS Linear Regression: Predicting Disruption Duration</div>
        <div class="card-desc">
            Classification only answers whether an outage will occur. Ordinary Least Squares (OLS) regression estimates how long 
            the satellite navigation outage will last in continuous minutes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    y_reg_test = pipeline['y_reg_test']
    reg_preds = np.maximum(0, reg_model.predict(pipeline['X_test']))
    r2 = r2_score(y_reg_test, reg_preds)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))

    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num">{r2*100:.1f}%</div>
            <div class="metric-label">R-Squared (Variance Explained)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Percentage of duration variance captured</div>
        </div>
        """, unsafe_allow_html=True)
    with rcol2:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num">{mae:.1f} min</div>
            <div class="metric-label">Mean Absolute Error (MAE)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Average error across all test days</div>
        </div>
        """, unsafe_allow_html=True)
    with rcol3:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-num">{rmse:.1f} min</div>
            <div class="metric-label">Root Mean Squared Error (RMSE)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Standard deviation of residual errors</div>
        </div>
        """, unsafe_allow_html=True)

    # Correlation Table
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Pearson Correlation with Disruption Duration</div>
        <div class="card-desc">
            Southward magnetic field (Bz) and solar wind speed show the strongest correlation with outage length.
        </div>
    </div>
    """, unsafe_allow_html=True)

    corr_series = pipeline['df'].corr()['outage_duration_minutes'].drop(['outage_duration_minutes', 'gps_disruption_target'])
    corr_df = pd.DataFrame({
        'Telemetry Feature': corr_series.index,
        'Correlation with Duration (Minutes)': corr_series.values
    }).sort_values(by='Correlation with Duration (Minutes)', ascending=True)

    fig_corr, ax_corr = plt.subplots(figsize=(8, 2.8))
    colors = ['#0284c7' if x > 0 else '#94a3b8' for x in corr_df['Correlation with Duration (Minutes)']]
    ax_corr.barh(corr_df['Telemetry Feature'], corr_df['Correlation with Duration (Minutes)'], color=colors, height=0.55)
    ax_corr.set_xlabel("Pearson Correlation Coefficient (r)")
    ax_corr.spines['top'].set_visible(False)
    ax_corr.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_corr)

# ------------------------------------------------------------------------------
# TAB 5: BENCHMARKS & AUDIT TRAIL
# ------------------------------------------------------------------------------
with tab5:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Model Benchmarks Across the Course Spectrum</div>
        <div class="card-desc">
            Comparison of candidate algorithms covered in class, from simple interpretable trees to ensemble methods.
        </div>
    </div>
    """, unsafe_allow_html=True)

    X_test_scaled = pipeline['scaler'].transform(pipeline['X_test'])
    models_dict = {
        "Constrained Decision Tree (Depth=4)": (pipeline['dt'], pipeline['X_test']),
        "Random Forest Ensemble (100 Trees)": (pipeline['rf'], pipeline['X_test']),
        "Gradient Boosted Trees (Sequential)": (pipeline['gb'], pipeline['X_test']),
        "Logistic Regression (Log-Odds Baseline)": (pipeline['lr'], X_test_scaled)
    }

    bench_rows = []
    for name, (m, test_data) in models_dict.items():
        acc = m.score(test_data, pipeline['y_test'])
        probs = m.predict_proba(test_data)[:, 1]
        auc = roc_auc_score(pipeline['y_test'], probs)
        bench_rows.append({
            "Model Architecture": name,
            "Test Accuracy": f"{acc*100:.2f}%",
            "ROC-AUC Score": f"{auc:.3f}",
            "Interpretability": "High (Rule-based)" if "Decision Tree" in name or "Logistic" in name else "Medium (Black-box ensemble)"
        })
    st.table(pd.DataFrame(bench_rows))

    # Feature Importance Plot
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Opening the Black Box: Gini Feature Importances</div>
        <div class="card-desc">
            Relative importance of each space weather feature in the Gradient Boosted Tree model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    importances = gb_model.feature_importances_
    feat_names = pipeline['features']
    sorted_idx = np.argsort(importances)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 2.8))
    ax_imp.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx], color='#0f766e', height=0.55)
    ax_imp.set_xlabel("Relative Importance (Gini Impurity Reduction)")
    ax_imp.spines['top'].set_visible(False)
    ax_imp.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_imp)

    # FAA Compliance Decision Log
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">FAA Compliance: Flight Dispatch Audit Record</div>
        <div class="card-desc">
            Permanent record of flight dispatch decisions, tracking human acceptance and discretionary overrides.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.decision_log:
        log_df = pd.DataFrame(st.session_state.decision_log)
        st.dataframe(log_df, use_container_width=True)
        csv_data = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Audit Log (.csv)",
            data=csv_data,
            file_name="flight_dispatch_audit_trail.csv",
            mime="text/csv"
        )
    else:
        st.caption("No flight certifications logged in this session yet. Use Tab 1 to record flight route decisions.")
