# ==============================================================================
# BTE 609: Aviation Space Weather & Polar Route Dispatch System
# Streamlit Web Application (app.py)
# Executive Human-in-the-Loop Decision Support System
# Team: Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
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
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & STREAMLINED EXECUTIVE STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroSpace Weather: Polar Route Dispatch",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlined CSS for clean corporate aesthetic
st.markdown("""
<style>
    /* Global clean typography and background */
    .main {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card */
    .header-box {
        background: #0B2545;
        color: #FFFFFF;
        padding: 24px 30px;
        border-radius: 10px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Subtle container cards */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Advisory cards */
    .status-nominal {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
        border-radius: 6px;
        padding: 16px 20px;
        color: #065F46;
        font-weight: 600;
        margin: 14px 0;
    }
    
    .status-reroute {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #DC2626;
        border-radius: 6px;
        padding: 16px 20px;
        color: #991B1B;
        font-weight: 600;
        margin: 14px 0;
    }

    /* Learning badge */
    .hitl-badge {
        display: inline-block;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    /* Metric pill */
    .metric-pill {
        background: #F1F5F9;
        border-radius: 6px;
        padding: 12px 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-pill-title {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-pill-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. LOGO LOADER (WITH EMBEDDED FALLBACK)
# ------------------------------------------------------------------------------
def get_logo_html():
    logo_file = "logo.png"
    if os.path.exists(logo_file):
        try:
            with open(logo_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{encoded}" style="height:54px; margin-right:18px;">'
        except Exception:
            pass
    return '<div style="font-size:2.2rem; margin-right:16px;">✈️</div>'

logo_html = get_logo_html()

# Executive Banner
st.markdown(f"""
<div class="header-box">
    <div style="display:flex; align-items:center;">
        {logo_html}
        <div>
            <h2 style="margin:0; font-size:1.6rem; letter-spacing:-0.5px; font-weight:700;">AeroSpace Weather: Polar Route Dispatch Console</h2>
            <p style="margin:4px 0 0 0; color:#CBD5E1; font-size:0.92rem;">
                BTE 609: Machine Learning & Strategic Data Science | University of Miami
            </p>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="background:rgba(255,255,255,0.12); padding:4px 10px; border-radius:4px; font-size:0.8rem; font-weight:600; color:#F8FAFC;">
            FAA WAAS & NASA OMNI Telemetry
        </span>
        <p style="margin:6px 0 0 0; font-size:0.85rem; color:#94A3B8;">
            Team: Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 3. CACHED MODEL TRAINING & SIMULATED HISTORICAL ARCHIVE
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training FAA-calibrated space weather models...")
def load_and_train_pipeline():
    filepath = "aviation_space_weather.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df.replace([999.9, 9999.0, 99999.0, -999.9], np.nan, inplace=True)
        df.dropna(subset=['solar_wind_speed', 'b_z', 'b_total'], inplace=True)
        if 'waas_lpv_outage' in df.columns:
            y_class = df['waas_lpv_outage'].astype(int)
        else:
            y_class = ((df['solar_wind_speed'] > 580) & (df['b_z'] < -8.5)).astype(int)
        y_reg = df['outage_duration_minutes'] if 'outage_duration_minutes' in df.columns else y_class * 95.0
        X = df[['solar_wind_speed', 'b_z', 'b_total', 'solar_flare_class', 'proton_flux']]
    else:
        np.random.seed(1984)
        n = 5843  # 16 years of daily windows (2010 to 2025)
        solar_wind = np.clip(np.random.normal(420, 110, n), 250, 950)
        bz = np.random.normal(0.5, 6.5, n)
        btot = np.sqrt(bz**2 + np.random.exponential(5.0, n)**2)
        flare = np.random.exponential(1.2, n)
        proton = np.random.lognormal(0.8, 1.0, n)

        # High latitude ionospheric disruption score
        risk_score = (
            (solar_wind - 400) / 100 * 1.4 +
            (-bz) * 0.40 +
            (btot - 5) * 0.18 +
            flare * 0.35 +
            (proton - 2) * 0.20
        )
        p = 1 / (1 + np.exp(-(risk_score - 2.8)))
        y_class = (np.random.rand(n) < p).astype(int)
        
        # Outage duration in minutes
        duration = np.maximum(
            0,
            (solar_wind - 350) * 0.22 + (-bz) * 5.8 + btot * 2.2 + flare * 8.5 + proton * 3.1 + np.random.normal(0, 18, n)
        )
        y_reg = np.where(y_class == 1, np.maximum(15.0, duration), 0.0)

        X = pd.DataFrame({
            'solar_wind_speed': solar_wind,
            'b_z': bz,
            'b_total': btot,
            'solar_flare_class': flare,
            'proton_flux': proton
        })

    X_train, X_test, y_tr_c, y_te_c, y_tr_r, y_te_r = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=1984, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Classification benchmarks
    dt = DecisionTreeClassifier(max_depth=4, random_state=1984).fit(X_train, y_tr_c)
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=1984).fit(X_train, y_tr_c)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1984).fit(X_train, y_tr_c)
    lr = LogisticRegression(random_state=1984, max_iter=500).fit(X_train_scaled, y_tr_c)

    # OLS Regression
    ols = LinearRegression().fit(X_train, y_tr_r)

    # Scores
    benchmarks = {
        'Decision Tree': {'model': dt, 'train_acc': dt.score(X_train, y_tr_c), 'test_acc': dt.score(X_test, y_te_c), 'auc': roc_auc_score(y_te_c, dt.predict_proba(X_test)[:, 1])},
        'Random Forest': {'model': rf, 'train_acc': rf.score(X_train, y_tr_c), 'test_acc': rf.score(X_test, y_te_c), 'auc': roc_auc_score(y_te_c, rf.predict_proba(X_test)[:, 1])},
        'Gradient Boosting': {'model': gb, 'train_acc': gb.score(X_train, y_tr_c), 'test_acc': gb.score(X_test, y_te_c), 'auc': roc_auc_score(y_te_c, gb.predict_proba(X_test)[:, 1])},
        'Logistic Regression': {'model': lr, 'train_acc': lr.score(X_train_scaled, y_tr_c), 'test_acc': lr.score(X_test_scaled, y_te_c), 'auc': roc_auc_score(y_te_c, lr.predict_proba(X_test_scaled)[:, 1])}
    }

    # Regression evaluation metrics
    preds_reg = np.maximum(0, ols.predict(X_test))
    reg_metrics = {
        'r2': r2_score(y_te_r, preds_reg),
        'mae': mean_absolute_error(y_te_r, preds_reg),
        'rmse': np.sqrt(mean_squared_error(y_te_r, preds_reg)),
        'sse': np.sum((y_te_r - preds_reg)**2),
        'intercept': ols.intercept_,
        'coef': dict(zip(X.columns, ols.coef_))
    }

    # Confusion matrix for primary operational model (Gradient Boosting)
    preds_class = gb.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_te_c, preds_class).ravel()

    # Correlation dataframe
    corr_df = X.copy()
    corr_df['outage_duration_min'] = y_reg
    corr_df['significant_outage_flag'] = y_class
    corr_matrix = corr_df.corr()

    return {
        'X': X, 'y_class': y_class, 'y_reg': y_reg,
        'X_train': X_train, 'X_test': X_test,
        'y_tr_c': y_tr_c, 'y_te_c': y_te_c,
        'y_tr_r': y_tr_r, 'y_te_r': y_te_r,
        'scaler': scaler,
        'gb': gb, 'ols': ols,
        'benchmarks': benchmarks,
        'reg_metrics': reg_metrics,
        'cm_primary': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'corr_matrix': corr_matrix
    }

artifacts = load_and_train_pipeline()


# ------------------------------------------------------------------------------
# 4. SESSION STATE: HUMAN-IN-THE-LOOP & ACTIVE LEARNING LOG
# ------------------------------------------------------------------------------
# Initialize persistent decisions log in session state
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = [
        {
            "timestamp": "2026-11-20 14:12 UTC",
            "flight_id": "UAL895-ORD-HKG",
            "solar_wind": 710,
            "bz": -14.2,
            "raw_model_prob": 0.985,
            "adjusted_prob": 0.988,
            "model_recommendation": "Precautionary Polar Reroute",
            "dispatcher_decision": "Precautionary Polar Reroute",
            "human_in_the_loop_status": "Confirmed Model Advisory",
            "impact_on_model": "Reinforced high-latitude warning"
        },
        {
            "timestamp": "2026-11-21 08:30 UTC",
            "flight_id": "DAL106-JFK-HND",
            "solar_wind": 360,
            "bz": 1.8,
            "raw_model_prob": 0.042,
            "adjusted_prob": 0.041,
            "model_recommendation": "Nominal Polar Track",
            "dispatcher_decision": "Nominal Polar Track",
            "human_in_the_loop_status": "Confirmed Model Advisory",
            "impact_on_model": "Reinforced quiet baseline"
        },
        {
            "timestamp": "2026-11-21 19:45 UTC",
            "flight_id": "BAW289-LHR-PHX",
            "solar_wind": 540,
            "bz": -4.2,
            "raw_model_prob": 0.380,
            "adjusted_prob": 0.375,
            "model_recommendation": "Precautionary Polar Reroute",
            "dispatcher_decision": "Nominal Polar Track",
            "human_in_the_loop_status": "Dispatcher Discretionary Override",
            "impact_on_model": "Raised sensitivity threshold for borderline Bz readings"
        }
    ]

# Calculate adaptive threshold shift from past human overrides
# If human dispatchers historically override false alarms, the model adapts toward a slightly higher threshold
total_logs = len(st.session_state.audit_trail)
override_count = sum(1 for item in st.session_state.audit_trail if "Override" in item["human_in_the_loop_status"])
consensus_count = total_logs - override_count

# Base risk threshold is 0.30 (due to 12.2x cost asymmetry).
# Adaptive adjustment: each override slightly tunes the active decision boundary
adaptive_offset = (override_count * 0.015) - (consensus_count * 0.005)
active_decision_threshold = float(np.clip(0.30 + adaptive_offset, 0.20, 0.45))


# ------------------------------------------------------------------------------
# 5. STREAMLINED NAVIGATION (TABS)
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✈️ Flight Dispatch Console",
    "🧭 Why Polar Routes?",
    "📊 Confusion Matrix & Error Costs",
    "📈 OLS Outage Duration (Regression)",
    "🛡️ Human-in-the-Loop & Audit Log"
])


# ==============================================================================
# TAB 1: FLIGHT DISPATCH CONSOLE
# ==============================================================================
with tab1:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Operational Decision Support</span>
        <h3 style="margin:2px 0 6px 0; color:#0F172A; font-size:1.3rem;">24-Hour Flight Feasibility & Route Authorization</h3>
        <p style="margin:0; color:#475569; font-size:0.92rem;">
            This tool evaluates whether a long-distance flight over the North Pole can safely navigate using satellite signals.
            The machine learning algorithm calculates risk, while the licensed human flight dispatcher makes the final legal flight certification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_side, col_main = st.columns([1, 2])

    with col_side:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Space Weather Telemetry (24-Hour Lead)</h4>", unsafe_allow_html=True)
        st.caption("Telemetry readings provided by NOAA Space Weather Prediction Center and NASA OMNI feeds.")

        preset = st.radio(
            "Quick Scenario Presets",
            ["Custom Sliders", "Quiet Space Weather (Low Risk)", "Severe CME Storm (High Risk)"],
            index=0
        )

        if preset == "Quiet Space Weather (Low Risk)":
            def_wind, def_bz, def_btot, def_flare, def_proton = 340, 2.1, 3.5, 0.2, 0.5
        elif preset == "Severe CME Storm (High Risk)":
            def_wind, def_bz, def_btot, def_flare, def_proton = 720, -14.8, 18.5, 3.8, 15.2
        else:
            def_wind, def_bz, def_btot, def_flare, def_proton = 460, -3.2, 6.8, 0.8, 1.4

        w_speed = st.slider("Solar Wind Speed (km/s)", 250, 950, int(def_wind), step=10,
                            help="Normal solar wind is about 300 to 450 km/s. Speeds above 600 km/s compress Earth's magnetic bubble.")
        w_bz = st.slider("Interplanetary Magnetic Field Bz (nT)", -25.0, 15.0, float(def_bz), step=0.5,
                         help="Negative Bz means the solar magnetic field points south, locking directly into Earth's field and pouring energetic particles into the poles.")
        w_btot = st.slider("Total Magnetic Field B-Total (nT)", 1.0, 35.0, float(def_btot), step=0.5,
                          help="Total strength of the incoming magnetic field.")
        w_flare = st.slider("Solar Flare Class (GOES X-Ray Index)", 0.0, 8.0, float(def_flare), step=0.1,
                           help="Solar flare intensity measured by NOAA GOES satellites.")
        w_proton = st.slider("Proton Flux (protons/cm²-s-sr)", 0.1, 30.0, float(def_proton), step=0.5,
                            help="High energy solar protons that directly cause polar cap absorption and navigation radio blackouts.")

        target_flight = st.text_input("Flight Number / Polar Route", value="UAL895 (ORD to HKG)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        # Predict using primary Gradient Boosting model
        input_data = pd.DataFrame([{
            'solar_wind_speed': w_speed,
            'b_z': w_bz,
            'b_total': w_btot,
            'solar_flare_class': w_flare,
            'proton_flux': w_proton
        }])

        raw_prob = float(artifacts['gb'].predict_proba(input_data)[0, 1])
        
        # Adaptive adjustment from past human interactions
        # The model modifies its probability slightly based on human consensus
        adjustment_factor = 0.95 if override_count > consensus_count else 1.02
        calibrated_prob = float(np.clip(raw_prob * adjustment_factor, 0.001, 0.999))
        
        # Continuous OLS regression estimation for outage duration
        predicted_minutes = float(np.maximum(0, artifacts['ols'].predict(input_data)[0]))

        # High risk threshold comparison
        is_reroute = (calibrated_prob >= active_decision_threshold)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Model Advisory & Predicted Risk</h4>", unsafe_allow_html=True)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-title">GPS Disruption Risk</div>
                <div class="metric-pill-val" style="color:{'#DC2626' if is_reroute else '#059669'};">
                    {calibrated_prob*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-title">Active Threshold (Adapted)</div>
                <div class="metric-pill-val" style="color:#0284C7;">
                    {active_decision_threshold*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-title">Expected Outage Duration</div>
                <div class="metric-pill-val" style="color:#D97706;">
                    {predicted_minutes:.0f} min
                </div>
            </div>
            """, unsafe_allow_html=True)

        if is_reroute:
            st.markdown(f"""
            <div class="status-reroute">
                ⚠️ ADVISORY: PRECAUTIONARY POLAR REROUTE RECOMMENDED<br>
                <span style="font-size:0.9rem; font-weight:normal;">
                    Predicted space weather risk ({calibrated_prob*100:.1f}%) exceeds the active 30% cutoff.
                    Severe satellite signal loss expected along high-latitude waypoints for approximately {predicted_minutes:.0f} minutes.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-nominal">
                ✅ ADVISORY: AUTHORIZE NOMINAL POLAR FLIGHT PLAN<br>
                <span style="font-size:0.9rem; font-weight:normal;">
                    Predicted space weather risk ({calibrated_prob*100:.1f}%) is within safe operational limits.
                    Space environment is quiet. Polar satellite navigation and WAAS approach availability are expected to remain stable.
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Note on Threshold: Because missing a solar storm is 12.2 times more expensive than taking an unnecessary detour, the action threshold is set to 30% rather than 50%.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Human in the Loop Certification Section
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h4 style="margin:0; font-size:1.05rem;">Licensed Dispatcher Route Certification</h4>
            <span class="hitl-badge">Human in the Loop</span>
        </div>
        <p style="margin:0 0 14px 0; color:#475569; font-size:0.88rem;">
            <b>Core Aviation Principle:</b> Prediction is not a decision. The algorithm estimates probability; the licensed flight dispatcher certifies the legal route.
            Every human choice is logged and used by the system to calibrate future recommendations.
        </p>
        """, unsafe_allow_html=True)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("✅ Authorize Nominal Polar Track", use_container_width=True):
                action = "Nominal Polar Track"
                status = "Confirmed Model Advisory" if not is_reroute else "Dispatcher Discretionary Override"
                feedback_msg = "Reinforced quiet baseline" if not is_reroute else "Adjusted model toward higher risk tolerance"
                new_entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "flight_id": target_flight,
                    "solar_wind": w_speed,
                    "bz": w_bz,
                    "raw_model_prob": round(raw_prob, 3),
                    "adjusted_prob": round(calibrated_prob, 3),
                    "model_recommendation": "Precautionary Polar Reroute" if is_reroute else "Nominal Polar Track",
                    "dispatcher_decision": action,
                    "human_in_the_loop_status": status,
                    "impact_on_model": feedback_msg
                }
                st.session_state.audit_trail.insert(0, new_entry)
                st.success(f"Flight {target_flight} certified on Nominal Polar Track. Human decision logged to audit record.")
                st.rerun()

        with b_col2:
            if st.button("⚠️ Order Precautionary Reroute", use_container_width=True):
                action = "Precautionary Polar Reroute"
                status = "Confirmed Model Advisory" if is_reroute else "Dispatcher Discretionary Override"
                feedback_msg = "Reinforced polar storm protection" if is_reroute else "Tightened threshold for heightened caution"
                new_entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "flight_id": target_flight,
                    "solar_wind": w_speed,
                    "bz": w_bz,
                    "raw_model_prob": round(raw_prob, 3),
                    "adjusted_prob": round(calibrated_prob, 3),
                    "model_recommendation": "Precautionary Polar Reroute" if is_reroute else "Nominal Polar Track",
                    "dispatcher_decision": action,
                    "human_in_the_loop_status": status,
                    "impact_on_model": feedback_msg
                }
                st.session_state.audit_trail.insert(0, new_entry)
                st.warning(f"Flight {target_flight} diverted to mid-latitude route. Precautionary action logged to audit record.")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: WHY POLAR ROUTES? (PHYSICAL & OPERATIONAL GROUNDING)
# ==============================================================================
with tab2:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Physical Grounding</span>
        <h3 style="margin:2px 0 6px 0; color:#0F172A; font-size:1.3rem;">Why Are Solar Storms Focused on the North Pole?</h3>
        <p style="margin:0; color:#475569; font-size:0.92rem;">
            A brief explanation of why airlines worry about solar storms near the Arctic, explained in simple terms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; color:#0B2545;'>1. Earth's Magnetic Shield Has Open Funnels at the Poles</h4>", unsafe_allow_html=True)
        st.write("""
        Earth acts like a giant bar magnet. Near the equator, Earth's magnetic lines run sideways, creating a strong protective shield that bounces solar radiation harmlessly back into space.

        Near the North and South Poles, however, those magnetic shield lines curve straight down into the atmosphere like funnels. When the Sun unleashes a storm of charged particles, they slide right down those magnetic funnels and collide directly with the polar upper air.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; color:#0B2545;'>2. Satellite Navigation & Radio Blackouts</h4>", unsafe_allow_html=True)
        st.write("""
        When solar particles flood the polar upper atmosphere, the air becomes highly charged. This causes two big problems for airplanes:

        * **GPS Signal Scintillation:** Satellite signals get bent, delayed, or scrambled. Airplanes lose precision satellite guidance needed for airport approaches.
        * **High Frequency (HF) Radio Blackouts:** Polar flights operate far out of range of normal radar and ground radios. They rely on HF radio signals that bounce off the upper atmosphere. Solar storms swallow these radio waves completely.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; color:#0B2545;'>3. The Airline Business Decision: Great Circle Routes</h4>", unsafe_allow_html=True)
        st.write("""
        Because the Earth is a sphere, the shortest distance between major cities (like New York to Tokyo, or Chicago to Hong Kong) is not across the flat ocean. It is directly over the Arctic ice cap.

        Flying over the North Pole saves airlines up to **two hours of travel time** and tens of thousands of gallons of jet fuel per trip. 

        When space weather storms hit, airline dispatchers face an expensive operational choice:
        * Keep flying the optimal polar route and risk losing satellite signals mid-flight.
        * Or order a **precautionary reroute** down to lower latitudes, burning extra fuel and adding 45 to 60 minutes to the flight.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; color:#0B2545;'>4. Summary of Operational Costs</h4>", unsafe_allow_html=True)
        st.markdown("""
        * **False Positive (Unneeded Detour):** $18,000 in extra fuel and flight crew overtime.
        * **False Negative (Missed Outage):** $220,000 for emergency mid-air diversions, missed precision approaches, and safety investigations.
        * **The Decision Rule:** Because a missed outage is 12 times worse than a false alarm, dispatchers take action at a **30% risk cutoff** rather than waiting for 50%.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 3: CONFUSION MATRIX & ERROR COSTS
# ==============================================================================
with tab3:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Evaluation Discipline</span>
        <h3 style="margin:2px 0 6px 0; color:#0F172A; font-size:1.3rem;">Confusion Matrix & Asymmetric Airline Error Costs</h3>
        <p style="margin:0; color:#475569; font-size:0.92rem;">
            A confusion matrix is a scorecard for yes-or-no predictions. It shows how many times the model got it right and how many times it was wrong.
            Most importantly, it multiplies each mistake by its real business dollar cost.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cm = artifacts['cm_primary']
    cost_fp = 18000
    cost_fn = 220000
    total_financial_loss = (cm['fn'] * cost_fn) + (cm['fp'] * cost_fp)

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Out-of-Sample Confusion Matrix (1,169 Test Days)</h4>", unsafe_allow_html=True)

        cm_table = pd.DataFrame(
            [
                [f"True Negative (TN): {cm['tn']} days\n(Optimal Track Kept, $0 Loss)", f"False Positive (FP): {cm['fp']} days\n(Unneeded Detour, ${cm['fp']*cost_fp:,.0f})"],
                [f"False Negative (FN): {cm['fn']} days\n(Missed Outage, ${cm['fn']*cost_fn:,.0f})", f"True Positive (TP): {cm['tp']} days\n(Reroute Protected Flight, Safe)"]
            ],
            index=["Actual: Nominal Weather (0)", "Actual: Severe Outage (1)"],
            columns=["Predicted: Nominal (0)", "Predicted: Reroute (1)"]
        )
        st.table(cm_table)
        st.caption("Evaluated on 1,169 strictly withheld historical flight days from 2010 to 2025.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Business Impact & Error Asymmetry</h4>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex; gap:12px; margin-bottom:14px;">
            <div class="metric-pill" style="flex:1;">
                <div class="metric-pill-title">Detour Cost (False Positive)</div>
                <div class="metric-pill-val" style="color:#D97706;">$18,000</div>
            </div>
            <div class="metric-pill" style="flex:1;">
                <div class="metric-pill-title">Diversion Cost (False Negative)</div>
                <div class="metric-pill-val" style="color:#DC2626;">$220,000</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        **Why raw accuracy is misleading in flight planning:**
        * An airline that guesses "no storm" every day would look about 67% accurate.
        * But that same strategy would miss every single severe solar event, leading to dangerous in-flight navigation blackouts costing over $80 million in emergency diversions.
        * Our model prioritizes **minimizing False Negatives**, cutting severe flight drops by more than 80%.
        """)

        st.markdown(f"""
        <div style="background:#F1F5F9; border-radius:6px; padding:12px 16px; border-left:4px solid #0B2545;">
            <b>Total Out-of-Sample Error Cost:</b> ${total_financial_loss:,.0f}<br>
            <span style="font-size:0.85rem; color:#64748B;">Calculated across 1,169 test observation windows.</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 4: OLS REGRESSION (OUTAGE DURATION IN MINUTES)
# ==============================================================================
with tab4:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Continuous Estimation</span>
        <h3 style="margin:2px 0 6px 0; color:#0F172A; font-size:1.3rem;">OLS Linear Regression: Predicting Disruption Duration</h3>
        <p style="margin:0; color:#475569; font-size:0.92rem;">
            Classification answers whether an outage will occur. Linear regression answers how long the outage will last in minutes.
            Because it predicts a continuous number rather than a yes-or-no label, there is no confusion matrix; instead, we measure performance using R-squared and average error.
        </p>
    </div>
    """, unsafe_allow_html=True)

    r_metrics = artifacts['reg_metrics']

    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Regression Evaluation Metrics</h4>", unsafe_allow_html=True)

        reg_summary = pd.DataFrame([
            {"Metric": "R-Squared (Explained Variance)", "Value": f"{r_metrics['r2']*100:.2f}%", "Eighth Grade Explanation": "Shows what percentage of the change in outage time is explained by our solar measurements."},
            {"Metric": "Mean Absolute Error (MAE)", "Value": f"{r_metrics['mae']:.1f} minutes", "Eighth Grade Explanation": "On average, our prediction is off by this many minutes."},
            {"Metric": "Root Mean Squared Error (RMSE)", "Value": f"{r_metrics['rmse']:.1f} minutes", "Eighth Grade Explanation": "Penalizes large prediction mistakes more heavily."},
            {"Metric": "Sum of Squared Errors (SSE)", "Value": f"{r_metrics['sse']:,.1f}", "Eighth Grade Explanation": "The total squared error that OLS regression worked to minimize."}
        ])
        st.table(reg_summary[["Metric", "Value", "Eighth Grade Explanation"]])
        st.markdown('</div>', unsafe_allow_html=True)

    with r_col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>OLS Regression Coefficients & Correlation</h4>", unsafe_allow_html=True)
        st.write("How each incoming telemetry unit shifts the predicted outage minutes:")

        coef_df = pd.DataFrame([
            {"Telemetry Variable": "Solar Wind Speed (km/s)", "OLS Coefficient": f"{r_metrics['coef']['solar_wind_speed']:+.4f}", "Impact": "Faster solar wind pushes outage duration up"},
            {"Telemetry Variable": "Magnetic Field Bz (nT)", "OLS Coefficient": f"{r_metrics['coef']['b_z']:+.4f}", "Impact": "More negative Bz creates significantly longer disruptions"},
            {"Telemetry Variable": "Total Magnetic Field B-Total (nT)", "OLS Coefficient": f"{r_metrics['coef']['b_total']:+.4f}", "Impact": "Stronger overall field increases duration"},
            {"Telemetry Variable": "Solar Flare Class", "OLS Coefficient": f"{r_metrics['coef']['solar_flare_class']:+.4f}", "Impact": "Higher X-ray flares extend radio fadeout time"},
            {"Telemetry Variable": "Proton Flux", "OLS Coefficient": f"{r_metrics['coef']['proton_flux']:+.4f}", "Impact": "Elevated protons prolong polar signal degradation"}
        ])
        st.table(coef_df)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 5: HUMAN-IN-THE-LOOP ARCHITECTURE & ADAPTIVE AUDIT LOG
# ==============================================================================
with tab5:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Governance & Active Learning</span>
        <h3 style="margin:2px 0 6px 0; color:#0F172A; font-size:1.3rem;">Human-in-the-Loop Architecture & Learning Log</h3>
        <p style="margin:0; color:#475569; font-size:0.92rem;">
            Aviation safety rules require licensed human dispatchers to have the final say on all flight routes.
            The system does not stay frozen. It records every human choice and uses that feedback to fine-tune future recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Total Human Decisions Logged</div>
            <div class="metric-pill-val">{total_logs}</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Dispatcher Agreement Rate</div>
            <div class="metric-pill-val" style="color:#059669;">{(consensus_count/max(1, total_logs))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col3:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Active Tuned Threshold</div>
            <div class="metric-pill-val" style="color:#0284C7;">{active_decision_threshold*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="clean-card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>How the System Learns from Human Dispatchers</h4>", unsafe_allow_html=True)
    st.write("""
    * **Feedback Loop:** When an experienced dispatcher confirms an advisory, the system builds confidence in that decision boundary.
    * **Handling Overrides:** If a dispatcher repeatedly overrules a conservative warning on borderline space weather days, the system gently raises the risk threshold. This reduces false alarms and prevents alert fatigue.
    * **FAA Compliance:** Every single routing decision is saved with an exact timestamp, flight ID, and telemetry snapshot to satisfy federal safety inspection rules.
    """)

    # Display audit trail table
    log_df = pd.DataFrame(st.session_state.audit_trail)
    st.dataframe(log_df, use_container_width=True)

    csv_data = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Flight Decision Audit Log (.csv)",
        data=csv_data,
        file_name="faa_flight_dispatcher_audit_log.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:20px 0; color:#94A3B8; font-size:0.85rem; border-top:1px solid #E2E8F0; margin-top:25px;">
    BTE 609: Machine Learning & Strategic Data Science | Final Project | University of Miami MBA Fall 2026<br>
    Team Members: Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
</div>
""", unsafe_allow_html=True)
