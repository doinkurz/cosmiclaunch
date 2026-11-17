# ==============================================================================
# BTE 609: Solar Storm & Space Weather Operational Risk Predictor
# Streamlit Web Application (app.py)
# Ready for instant deployment on Streamlit Community Cloud via GitHub
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
from sklearn.metrics import confusion_matrix, roc_auc_score

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Space Weather Launch Risk Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0052cc;
        margin-bottom: 10px;
    }
    .advisory-go {
        background-color: #d4edda;
        color: #155724;
        padding: 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.25rem;
        border-left: 6px solid #28a745;
    }
    .advisory-hold {
        background-color: #f8d7da;
        color: #721c24;
        padding: 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.25rem;
        border-left: 6px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# CACHED MODEL TRAINING PIPELINE
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training benchmark models on space weather archive...")
def train_space_weather_models():
    filepath = "historical_space_weather.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        column_map = {
            'V': 'solar_wind_speed', 'flow_speed': 'solar_wind_speed', 'speed': 'solar_wind_speed',
            'BZ': 'b_z', 'Bz': 'b_z', 'BZ_GSM': 'b_z', 'bz_gsm': 'b_z',
            'BT': 'b_total', 'B_total': 'b_total', 'Bt': 'b_total',
            'N': 'proton_density', 'density': 'proton_density', 'np': 'proton_density',
            'T': 'proton_temp', 'temp': 'proton_temp',
            'Kp': 'kp_index', 'Dst': 'dst_index'
        }
        df.rename(columns=column_map, inplace=True)
        df.replace([999.9, 9999.0, 99999.0, -999.9, -9999.0], np.nan, inplace=True)
        df.dropna(subset=['solar_wind_speed', 'b_z', 'b_total'], inplace=True)

        if 'kp_index' in df.columns:
            df['severe_storm_target'] = (df['kp_index'] >= 6).astype(int)
        elif 'dst_index' in df.columns:
            df['severe_storm_target'] = (df['dst_index'] <= -100).astype(int)
        else:
            df['severe_storm_target'] = ((df['solar_wind_speed'] > 600) & (df['b_z'] < -10)).astype(int)
    else:
        np.random.seed(1984)
        n_samples = 6000
        solar_wind = np.clip(np.random.normal(loc=420, scale=110, size=n_samples), 250, 950)
        b_z = np.random.normal(loc=0.5, scale=6.5, size=n_samples)
        b_total = np.sqrt(b_z**2 + np.random.exponential(scale=5.0, size=n_samples)**2)
        proton_density = np.random.lognormal(mean=1.5, sigma=0.6, size=n_samples)
        proton_temp = np.random.lognormal(mean=11.2, sigma=0.5, size=n_samples)

        storm_score = (
            (solar_wind - 400) / 100 * 1.5 +
            (-b_z) * 0.45 +
            (b_total - 5) * 0.20 +
            (proton_density - 5) * 0.15
        )
        storm_prob = 1 / (1 + np.exp(-(storm_score - 2.5)))
        target = (np.random.rand(n_samples) < storm_prob).astype(int)

        df = pd.DataFrame({
            'solar_wind_speed': solar_wind,
            'b_z': b_z,
            'b_total': b_total,
            'proton_density': proton_density,
            'proton_temp': proton_temp,
            'severe_storm_target': target
        })

    for col in ['proton_density', 'proton_temp']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    features = ['solar_wind_speed', 'b_z', 'b_total', 'proton_density', 'proton_temp']
    X = df[features]
    y = df['severe_storm_target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1984, stratify=y
    )

    # 1. Decision Tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=1984)
    dt.fit(X_train, y_train)

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=1984)
    rf.fit(X_train, y_train)

    # 3. Gradient Boosted Trees
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1984)
    gb.fit(X_train, y_train)

    models = {"Decision Tree": dt, "Random Forest": rf, "Gradient Boosted Tree": gb}
    metrics = {}
    for name, m in models.items():
        metrics[name] = {
            "Train Acc": m.score(X_train, y_train),
            "Test Acc": m.score(X_test, y_test),
            "ROC-AUC": roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        }

    return gb, models, metrics, features, X_test, y_test


gb_model, models, model_metrics, feature_names, X_test, y_test = train_space_weather_models()

# Initialize session state for persistent flight director logs
if "audit_log" not in st.session_state:
    if os.path.exists("flight_director_decision_log.csv"):
        st.session_state.audit_log = pd.read_csv("flight_director_decision_log.csv")
    else:
        st.session_state.audit_log = pd.DataFrame(columns=[
            'timestamp', 'solar_wind_speed', 'b_z', 'b_total', 'proton_density',
            'proton_temp', 'storm_probability', 'advisory_action',
            'flight_director_decision', 'override_flag', 'decision_status', 'operational_notes'
        ])


# ------------------------------------------------------------------------------
# SIDEBAR: LIVE TELEMETRY INPUT CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Live Telemetry Controls")
st.sidebar.caption("Simulate real-time NOAA SWPC solar sensor feeds:")

wind_input = st.sidebar.slider("Solar Wind Velocity (km/s)", min_value=250, max_value=950, value=580, step=10)
bz_input = st.sidebar.slider("Interplanetary Bz Vector (nT)", min_value=-25.0, max_value=15.0, value=-8.5, step=0.5)
btot_input = st.sidebar.slider("Total Magnetic Field B_tot (nT)", min_value=1.0, max_value=35.0, value=14.0, step=0.5)
density_input = st.sidebar.slider("Proton Density (N/cm³)", min_value=1.0, max_value=40.0, value=8.5, step=0.5)
temp_input = st.sidebar.slider("Proton Temperature (Kelvin)", min_value=10000, max_value=500000, value=120000, step=10000)

st.sidebar.markdown("---")
threshold_input = st.sidebar.slider("Operational Risk Threshold", min_value=0.10, max_value=0.60, value=0.35, step=0.05,
                                    help="Default 35%: Lower than 50% due to 166:1 asymmetric cost ratio.")
mission_notes = st.sidebar.text_input("Mission / Payload Reference", value="Starlink Group 6-18")


# ------------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# ------------------------------------------------------------------------------
st.title("🚀 Solar Storm & Cosmic Weather Operational Risk Predictor")
st.markdown("**BTE 609 Final Team Project** | *Decision Support System for Commercial Orbital Launch*")

tab_ops, tab_benchmarks, tab_economics, tab_audit = st.tabs([
    "🎯 Operational Flight Console",
    "📊 Model Benchmarking",
    "💰 Cost Arithmetic ($50M vs $300k)",
    "📜 Flight Director Audit Trail"
])

# ----------------- TAB 1: OPERATIONAL FLIGHT CONSOLE -----------------
with tab_ops:
    # Model Inference
    input_df = pd.DataFrame([{
        'solar_wind_speed': wind_input,
        'b_z': bz_input,
        'b_total': btot_input,
        'proton_density': density_input,
        'proton_temp': temp_input
    }])
    storm_prob = gb_model.predict_proba(input_df)[0, 1]
    is_hold = storm_prob >= threshold_input

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1.5])
    with col_m1:
        st.metric("Severe Storm Probability", f"{storm_prob * 100:.1f}%",
                  delta=f"{(storm_prob - threshold_input) * 100:+.1f}% vs Cutoff",
                  delta_color="inverse")
    with col_m2:
        st.metric("Operational Cutoff", f"{threshold_input * 100:.0f}%", "Asymmetric Safety Rule")
    with col_m3:
        if is_hold:
            st.markdown("""
            <div class="advisory-hold">
                ⚠️ ADVISORY: HOLD / DELAY LAUNCH<br>
                <small style="font-size:0.85rem">High ionospheric drag & orbital decay hazard detected.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="advisory-go">
                ✅ ADVISORY: GO FOR LAUNCH<br>
                <small style="font-size:0.85rem">Cosmic weather nominal. Safe for orbital insertion.</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Human-in-the-Loop Flight Director Confirmation")
    st.caption("Core Course Maxim: **'Prediction is not a decision.'** The model advises; human judgment decides.")

    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    decision_submitted = None

    with c_btn1:
        if st.button("CONFIRM: GO FOR LAUNCH", type="primary", use_container_width=True):
            decision_submitted = "GO"
    with c_btn2:
        if st.button("CONFIRM: NO GO (24-HR HOLD)", use_container_width=True):
            decision_submitted = "NO GO (HOLD)"

    if decision_submitted:
        human_hold = (decision_submitted == "NO GO (HOLD)")
        if human_hold and not is_hold:
            status = "OVERRIDE: Human Conservative Hold"
            override = True
        elif not human_hold and is_hold:
            status = "OVERRIDE: Human Accepted Risk (Go)"
            override = True
        else:
            status = "ACCEPTED: Human Confirmed Model"
            override = False

        new_entry = {
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            'solar_wind_speed': wind_input,
            'b_z': bz_input,
            'b_total': btot_input,
            'proton_density': density_input,
            'proton_temp': temp_input,
            'storm_probability': round(float(storm_prob), 4),
            'advisory_action': "HOLD / DELAY" if is_hold else "GO",
            'flight_director_decision': decision_submitted,
            'override_flag': override,
            'decision_status': status,
            'operational_notes': mission_notes
        }
        st.session_state.audit_log = pd.concat([st.session_state.audit_log, pd.DataFrame([new_entry])], ignore_index=True)
        st.session_state.audit_log.to_csv("flight_director_decision_log.csv", index=False)
        st.success(f"Decision Logged: **{decision_submitted}** ({status}) for payload **{mission_notes}**")

# ----------------- TAB 2: MODEL BENCHMARKING -----------------
with tab_benchmarks:
    st.subheader("Candidate Model Comparison Across Course Modules")
    st.caption("Benchmarking Decision Tree (interpretable), Random Forest (bagging), and Gradient Boosted Trees (boosting).")

    bench_df = pd.DataFrame(model_metrics).T
    bench_df["Train Acc"] = bench_df["Train Acc"].map(lambda x: f"{x * 100:.2f}%")
    bench_df["Test Acc"] = bench_df["Test Acc"].map(lambda x: f"{x * 100:.2f}%")
    bench_df["ROC-AUC"] = bench_df["ROC-AUC"].map(lambda x: f"{x:.3f}")
    st.table(bench_df)

    st.subheader("Interrogating the Black Box: Telemetry Feature Importances")
    importances = gb_model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(range(len(indices)), importances[indices], color='#0052cc', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Relative Importance Score (Gini Impurity Reduction)")
    ax.set_title("Feature Importances in Gradient Boosted Ensemble")
    st.pyplot(fig)

# ----------------- TAB 3: COST ARITHMETIC -----------------
with tab_economics:
    st.subheader("Translating ML Metrics into Business Capital")
    st.markdown("""
    * **False Negative (Missed Storm)**: Satellite burn-up / upper atmospheric orbital loss: **$50,000,000**
    * **False Positive (False Alarm)**: 24-hr launch range scrub and propellant boil-off: **$300,000**
    * **Asymmetry Ratio**: Missing a storm is **166.7x** more damaging than a 24-hour delay.
    """)

    cost_results = []
    for name, m in models.items():
        preds = m.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        loss = (fn * 50_000_000) + (fp * 300_000)
        cost_results.append({
            "Algorithm": name,
            "Missed Storms (FN)": fn,
            "False Alarms (FP)": fp,
            "Net Financial Error Cost": f"${loss:,.0f}"
        })
    st.dataframe(pd.DataFrame(cost_results), use_container_width=True)

# ----------------- TAB 4: AUDIT TRAIL -----------------
with tab_audit:
    st.subheader("Persistent Flight Director Decision Log")
    st.caption("Human audit records stored across sessions to satisfy FAA compliance and prevent model drift.")
    if len(st.session_state.audit_log) > 0:
        st.dataframe(st.session_state.audit_log.tail(15), use_container_width=True)
        csv_data = st.session_state.audit_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Decision Audit Log (.csv)", csv_data, "flight_director_decision_log.csv", "text/csv")
    else:
        st.info("No flight decisions logged yet. Use the Operational Flight Console tab to log decisions.")
