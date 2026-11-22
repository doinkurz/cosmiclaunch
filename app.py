import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="CosmicLaunch | Aviation GPS Storm Risk Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2630;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #2d3748;
    }
    .status-reroute {
        background-color: #742a2a;
        color: #fff5f5;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        border-left: 6px solid #e53e3e;
    }
    .status-nominal {
        background-color: #22543d;
        color: #f0fff4;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        border-left: 6px solid #38a169;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# STEP 1: CACHED DATA & MODEL TRAINING PIPELINE
# ==============================================================================
@st.cache_data
def load_and_train_models(random_seed=1984):
    np.random.seed(random_seed)
    n_samples = 5843  # ~16 years (2010–2025 daily horizons)

    # Physical Space Weather Predictors (24-hour lead time)
    solar_wind_speed = np.clip(np.random.normal(loc=420, scale=110, size=n_samples), 250, 950)
    b_z = np.random.normal(loc=0.5, scale=6.5, size=n_samples)
    b_total = np.sqrt(b_z**2 + np.random.exponential(scale=5.0, size=n_samples)**2)
    solar_flare_class = np.random.exponential(scale=1.2, size=n_samples)
    proton_flux = np.random.lognormal(mean=0.8, sigma=1.0, size=n_samples)

    # Physical target: FAA WAAS LPV degradation >= 15 min
    ionospheric_risk = (
        (solar_wind_speed - 400) / 100 * 1.4 +
        (-b_z) * 0.40 +
        (b_total - 5) * 0.18 +
        solar_flare_class * 0.35 +
        (proton_flux - 2) * 0.20
    )
    prob = 1 / (1 + np.exp(-(ionospheric_risk - 2.8)))
    y_class = (np.random.rand(n_samples) < prob).astype(int)

    # Continuous target: Outage duration in minutes
    base_duration = (
        0.18 * solar_wind_speed +
        -7.5 * b_z +
        3.2 * b_total +
        12.0 * solar_flare_class +
        5.5 * proton_flux - 35
    )
    y_reg = np.where(y_class == 1, np.clip(base_duration + np.random.normal(0, 20, size=n_samples), 15, 360), 0.0)

    df = pd.DataFrame({
        'solar_wind_speed': solar_wind_speed,
        'b_z': b_z,
        'b_total': b_total,
        'solar_flare_class': solar_flare_class,
        'proton_flux': proton_flux,
        'gps_disruption_target': y_class,
        'outage_duration_minutes': y_reg
    })

    X = df[['solar_wind_speed', 'b_z', 'b_total', 'solar_flare_class', 'proton_flux']]
    
    # Stratified split to prevent leakage
    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=random_seed, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Classification Models
    dt = DecisionTreeClassifier(max_depth=4, random_state=random_seed)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=random_seed)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_seed)
    gb.fit(X_train, y_train)

    lr = LogisticRegression(random_state=random_seed, max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # Regression Model (OLS Continuous Outage Duration)
    ols = LinearRegression()
    ols.fit(X_train, y_reg_train)

    return df, X_test, y_test, y_reg_test, dt, rf, gb, lr, ols, scaler

df, X_test, y_test, y_reg_test, dt_model, rf_model, gb_model, lr_model, ols_model, scaler = load_and_train_models()

# ==============================================================================
# HEADER & TEAM ROSTER
# ==============================================================================
st.title("✈️ CosmicLaunch: Aviation GPS & Space Weather Predictor")
st.markdown("""
**Course:** BTE 609 — Machine Learning & Artificial Intelligence (MBA Fall 2026)  
**Project Team:** Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer  
**Operational Scope:** 24-Hour Trans-Polar Flight Planning & FAA WAAS Precision Navigation Availability
""")

st.info("💡 **Core Managerial Discipline:** Prediction is not a decision. Machine learning models generate probabilistic risk forecasts; the licensed Airline Flight Dispatcher retains operational command.")

# ==============================================================================
# TABS INTERFACE
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛫 Flight Dispatch Console", 
    "📊 Confusion Matrix & Error Costs", 
    "📈 OLS Regression Matrix", 
    "🔬 Model Benchmarks & Importances", 
    "📋 Compliance Audit Log"
])

# ------------------------------------------------------------------------------
# TAB 1: OPERATIONAL FLIGHT DISPATCH CONSOLE
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Real-Time Telemetry Input & Flight Route Evaluation")
    st.markdown("Enter 24-hour forecasted space weather telemetry from NOAA SWPC / NASA OMNI to evaluate polar flight feasibility.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🛰️ Upstream Space Telemetry")
        wind = st.slider("Solar Wind Velocity (km/s)", min_value=250.0, max_value=950.0, value=480.0, step=10.0,
                         help="Speeds > 550 km/s indicate coronal mass ejection or high-speed solar stream.")
        bz = st.slider("Interplanetary Magnetic Field Vector Bz (nT)", min_value=-25.0, max_value=20.0, value=-4.5, step=0.5,
                       help="Negative (southward) Bz cancels Earth's shielding, injecting particles directly into ionosphere.")
        btot = st.slider("Total Magnetic Field Strength B_total (nT)", min_value=1.0, max_value=35.0, value=9.0, step=0.5)
        flare = st.slider("Solar X-Ray Flare Class (Goes Index)", min_value=0.0, max_value=8.0, value=1.2, step=0.1,
                          help="Measures ionizing flux causing immediate polar radio blackout (D-region absorption).")
        proton = st.slider("High-Energy Proton Flux (>10 MeV)", min_value=0.1, max_value=50.0, value=2.5, step=0.5)

    input_df = pd.DataFrame([{
        'solar_wind_speed': wind,
        'b_z': bz,
        'b_total': btot,
        'solar_flare_class': flare,
        'proton_flux': proton
    }])

    # Model Inferences
    risk_prob = gb_model.predict_proba(input_df)[0, 1]
    est_duration = max(0.0, float(ols_model.predict(input_df)[0]))

    with col2:
        st.markdown("### 🧭 Dispatch Operational Recommendation")
        
        cutoff = 0.30  # 30% risk threshold due to 12.2x cost asymmetry
        
        m1, m2 = st.columns(2)
        m1.metric("GPS Disruption Probability", f"{risk_prob*100:.1f}%", delta="Precautionary Action Cutoff: 30%", delta_color="inverse")
        m2.metric("Expected Outage Duration", f"{est_duration:.1f} min", delta="OLS Continuous Estimate")

        st.markdown("---")
        if risk_prob >= cutoff:
            st.markdown(f"""
            <div class="status-reroute">
                ⚠️ ADVISORY: PRECAUTIONARY POLAR REROUTE / HOLD LPV APPROACH<br>
                <small>High probability of severe ionospheric scintillation and loss of FAA WAAS precision service.</small>
            </div>
            """, unsafe_allow_html=True)
            st.error("Action: Route trans-polar flights via lower-latitude oceanic tracks. Advise flight crew of degraded satellite navigation.")
        else:
            st.markdown(f"""
            <div class="status-nominal">
                ✅ ADVISORY: AUTHORIZE NOMINAL POLAR FLIGHT TRACK<br>
                <small>Space weather conditions within acceptable operational thresholds.</small>
            </div>
            """, unsafe_allow_html=True)
            st.success("Action: Authorize flight plan as filed. Normal WAAS LPV approach and polar navigation expected.")

        st.markdown("### ✍️ Dispatcher Certification & Sign-Off")
        flight_id = st.text_input("Flight Number / Call Sign", value="DAL106-JFK-HND")
        
        c_act1, c_act2 = st.columns(2)
        if c_act1.button("CERTIFY: NOMINAL POLAR TRACK", use_container_width=True):
            st.session_state.setdefault('audit_log', []).append({
                'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                'flight_id': flight_id,
                'wind': wind,
                'bz': bz,
                'risk_prob': f"{risk_prob*100:.1f}%",
                'action': "NOMINAL TRACK",
                'audit': "ACCEPTED: Dispatcher Confirmed Safe" if risk_prob < cutoff else "OVERRIDE: Dispatcher Accepted Disruption Risk"
            })
            st.toast("✅ Flight dispatch recorded to FAA audit log.")

        if c_act2.button("CERTIFY: PRECAUTIONARY REROUTE", use_container_width=True):
            st.session_state.setdefault('audit_log', []).append({
                'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                'flight_id': flight_id,
                'wind': wind,
                'bz': bz,
                'risk_prob': f"{risk_prob*100:.1f}%",
                'action': "PRECAUTIONARY REROUTE",
                'audit': "ACCEPTED: Dispatcher Confirmed Reroute" if risk_prob >= cutoff else "OVERRIDE: Precautionary Discretionary Detour"
            })
            st.toast("⚠️ Precautionary polar reroute recorded to FAA audit log.")

# ------------------------------------------------------------------------------
# TAB 2: CONFUSION MATRIX & ASYMMETRIC AIRLINE COST MATRIX
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Confusion Matrix & The Arithmetic of Asymmetric Error Costs")
    st.markdown("""
    Standard accuracy fails in high-stakes aviation decisions. Because the operational and financial impact of errors is severely asymmetric, we convert the **Confusion Matrix** into real dollar costs.
    """)

    COST_FN = 220_000   # Missed Outage: Mid-air diversion, emergency holding, lost LPV approach
    COST_FP = 18_000    # False Alarm: Unneeded 60-min trans-polar detour & excess fuel burn

    preds_gb = gb_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_gb).ravel()
    net_cost = (fn * COST_FN) + (fp * COST_FP)

    c_box1, c_box2, c_box3 = st.columns(3)
    c_box1.metric("Missed Outage Cost (False Negative)", f"${COST_FN:,.0f}", "Severe In-Flight Hazard")
    c_box2.metric("Unneeded Reroute Cost (False Positive)", f"${COST_FP:,.0f}", "Excess Jet Fuel Burn")
    c_box3.metric("Cost Asymmetry Ratio", f"{COST_FN/COST_FP:.1f}x", "FN is 12.2x more costly")

    st.markdown("### Gradient Boosted Ensemble Confusion Matrix (1,169 Test Days)")
    cm_df = pd.DataFrame({
        "Predicted: Nominal Track (0)": [f"True Negative (TN): {tn}", f"False Negative (FN): {fn}"],
        "Predicted: Precautionary Reroute (1)": [f"False Positive (FP): {fp}", f"True Positive (TP): {tp}"]
    }, index=["Actual: Nominal Space Weather (0)", "Actual: FAA WAAS Outage (1)"])

    st.table(cm_df)

    st.markdown(f"""
    * **True Positives ({tp} flights)**: Successfully rerouted away from dangerous polar ionospheric scintillation.
    * **True Negatives ({tn} flights)**: Maintained optimal polar flight tracks, saving millions in fuel.
    * **False Positives ({fp} flights)**: Precautionary detours where storms did not materialize. Total cost penalty: **${fp*COST_FP:,.0f}**.
    * **False Negatives ({fn} flights)**: Unwarned in-flight navigation degradation forcing holding patterns and diversions. Total cost penalty: **${fn*COST_FN:,.0f}**.
    * **Net Financial Disruption Penalty**: **${net_cost:,.0f}**.
    """)

# ------------------------------------------------------------------------------
# TAB 3: OLS REGRESSION EVALUATION MATRIX
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("OLS Regression Matrix (Predicting Continuous Outage Minutes)")
    st.markdown("""
    While classification answers *whether* an outage will occur, **OLS Linear Regression** answers *how severe* it will be in continuous minutes.
    """)

    y_reg_pred = ols_model.predict(X_test)
    y_reg_pred_clipped = np.clip(y_reg_pred, 0, None)

    r2 = r2_score(y_reg_test, y_reg_pred_clipped)
    mae = mean_absolute_error(y_reg_test, y_reg_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_clipped))
    sse = np.sum((y_reg_test - y_reg_pred_clipped)**2)

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("R-Squared (Explained Variance)", f"{r2*100:.2f}%", "Proportion of total variation explained")
    r_col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} min", "Average prediction error")
    r_col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} min", "Penalty on large deviations")
    r_col4.metric("Sum of Squared Errors (SSE)", f"{sse:,.0f}", "Total squared residuals")

    st.markdown("### Feature Correlation Matrix (Pearson r)")
    corr = df.corr()
    st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None).format("{:.3f}"), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: BENCHMARKING & FEATURE IMPORTANCES
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Multi-Model Benchmarks Across the Accuracy-Interpretability Spectrum")
    
    models = {
        "Constrained Decision Tree (Depth=4)": (dt_model, X_test),
        "Random Forest Ensemble (100 Trees)": (rf_model, X_test),
        "Gradient Boosted Trees (Sequential)": (gb_model, X_test),
        "Logistic Regression (Log-Odds Baseline)": (lr_model, scaler.transform(X_test))
    }

    benchmarks = []
    for name, (mod, X_eval) in models.items():
        preds = mod.predict(X_eval)
        prob_eval = mod.predict_proba(X_eval)[:, 1]
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test, preds).ravel()
        benchmarks.append({
            "Model Architecture": name,
            "Test Accuracy": f"{np.mean(preds == y_test)*100:.2f}%",
            "ROC-AUC Score": f"{roc_auc_score(y_test, prob_eval):.3f}",
            "Missed Outages (FN)": t_fn,
            "Unneeded Detours (FP)": t_fp,
            "Total Operational Cost": f"${(t_fn*COST_FN + t_fp*COST_FP):,.0f}"
        })

    st.dataframe(pd.DataFrame(benchmarks), use_container_width=True)

    st.markdown("### Opening the Black Box: Gini Feature Importances")
    importances = gb_model.feature_importances_
    features = ['Solar Wind Speed', 'IMF Bz Vector', 'Total IMF (Btot)', 'Solar Flare Class', 'Proton Flux']
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(range(len(idx)), importances[idx], color='#1b4f72', align='center')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx])
    ax.set_xlabel("Relative Predictive Importance (Gini Impurity Reduction)")
    ax.set_title("Ranked Physical Telemetry Drivers of Navigation Failure")
    st.pyplot(fig)

# ------------------------------------------------------------------------------
# TAB 5: COMPLIANCE AUDIT LOG
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("FAA Flight Dispatch Decision Audit Trail")
    st.markdown("All human dispatcher certifications and system advisory recommendations are retained for regulatory compliance.")

    if 'audit_log' in st.session_state and len(st.session_state['audit_log']) > 0:
        log_df = pd.DataFrame(st.session_state['audit_log'])
        st.dataframe(log_df, use_container_width=True)
        csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Flight Decision Audit Log (.csv)",
            data=csv,
            file_name="faa_flight_dispatch_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No flight dispatch certifications logged in this session yet. Test a flight route decision in Tab 1 to generate audit records.")
