import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from fpdf import FPDF

# ===============================
# CONFIG
# ===============================
st.set_page_config("Security Risk Dashboard", layout="wide")

# ===============================
# CONSTANTS
# ===============================
QUAL_MAPPING = {
    "Poor": 20,
    "Fair": 50,
    "Good": 80,
    "Excellent": 100
}

RISK_LABELS = [
    "Unauthorized Access",
    "Insider Threat",
    "Emergency Failure",
    "Perimeter Breach"
]

# ===============================
# UTILITIES
# ===============================
def map_score(v):
    return QUAL_MAPPING[v] if isinstance(v, str) else float(v)

# ===============================
# RULE-BASED SCORING
# ===============================
def compute_scores(data):
    weights = {
        "Physical Security": 0.25,
        "Access Control": 0.30,
        "Personnel": 0.15,
        "Incident History": 0.20,
        "Emergency Preparedness": 0.10
    }

    category_scores = {}
    contributions = {}

    for cat, items in data.items():
        scores = [map_score(v) for v in items.values()]
        avg = np.mean(scores)
        category_scores[cat] = round(avg, 2)
        contributions[cat] = round(avg * weights[cat], 2)

    overall = round(sum(contributions.values()), 2)
    return category_scores, contributions, overall

# ===============================
# ML FEATURES (MATCH TRAINING)
# ===============================
def build_ml_features(data):
    """
    Converts rule-based input into ML feature format
    """
    return pd.DataFrame([{
        "size_employees": 580,
        "daily_visitors": 60,
        "facility_area_sqm": 22000,

        "cctv_coverage_pct": map_score(data["Physical Security"]["CCTV Coverage %"]),
        "cctv_functional_pct": map_score(data["Physical Security"]["CCTV Functionality %"]),
        "perimeter_cond_num": map_score(data["Physical Security"]["Perimeter Condition"]),
        "recording_sys_num": 30,
        "exterior_light_num": map_score(data["Physical Security"]["Lighting Quality"]),
        "interior_light_num": 70,

        "parking_security": 1,
        "total_guards": 12,
        "guard_to_area_ratio_per_1000sqm": 12 / 22,

        "training_frequency_years": 2,
        "background_check_num": map_score(data["Personnel"]["Background Checks"]),
        "turnover_rate_pct": 60,

        "documentation_quality_num": map_score(data["Incident History"]["Documentation Quality"]),
        "avg_response_time_min": 12,

        "communication_score": map_score(data["Emergency Preparedness"]["Communication System"]),
        "emergency_plan_flag": 0,
        "drill_frequency_per_year": 0
    }])

# ===============================
# SHAP SAFE HANDLER
# ===============================
def get_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    return shap_vals[1][0]  # class 1, first row

# ===============================
# PDF REPORT
# ===============================
def generate_pdf(category_scores, overall, shap_img=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Security Risk Assessment Report", ln=True, align="C")
    pdf.ln(5)
    pdf.cell(200, 10, f"Overall Risk Score: {overall}/100", ln=True)

    pdf.ln(5)
    for k, v in category_scores.items():
        pdf.cell(200, 8, f"{k}: {v}", ln=True)

    if shap_img:
        pdf.add_page()
        pdf.image(shap_img, w=180)

    path = "security_risk_report.pdf"
    pdf.output(path)
    return path

# ===============================
# ANOMALY ENGINE
# ===============================
BASELINES = {
    "Incident Severity": (40, 10),
    "Unauthorized Access": (35, 10),
    "CCTV Uptime": (90, 5),
    "After Hours": (65, 10)
}

def run_anomaly_engine(data):
    alerts = []

    current = {
        "Incident Severity": data["Incident History"]["Incident Severity Score"],
        "Unauthorized Access": data["Incident History"]["Incident Types Score"],
        "CCTV Uptime": data["Physical Security"]["CCTV Functionality %"],
        "After Hours": map_score(data["Access Control"]["After-Hours Security"])
    }

    for k, v in current.items():
        mean, std = BASELINES[k]
        z = (v - mean) / std
        if abs(z) >= 2:
            alerts.append((k, "HIGH", round(z, 2)))
        elif abs(z) >= 1.3:
            alerts.append((k, "MEDIUM", round(z, 2)))

    return alerts

# ===============================
# STREAMLIT UI
# ===============================
st.title("🔐 Security Risk Dashboard")

# ---- INPUTS ----
st.sidebar.header("Facility Inputs")

physical = {
    "Perimeter Condition": st.sidebar.selectbox("Perimeter Condition", QUAL_MAPPING),
    "CCTV Coverage %": st.sidebar.slider("CCTV Coverage %", 0, 100, 70),
    "CCTV Functionality %": st.sidebar.slider("CCTV Functionality %", 0, 100, 85),
    "Lighting Quality": st.sidebar.selectbox("Lighting Quality", QUAL_MAPPING)
}

access = {
    "After-Hours Security": st.sidebar.selectbox("After-Hours Security", QUAL_MAPPING)
}

personnel = {
    "Guard Count Ratio Score": st.sidebar.slider("Guard Adequacy Score", 0, 100, 70),
    "Background Checks": st.sidebar.selectbox("Background Checks", QUAL_MAPPING)
}

incident = {
    "Incident Severity Score": st.sidebar.slider("Incident Severity", 0, 100, 30),
    "Incident Types Score": st.sidebar.slider("Unauthorized Access Severity", 0, 100, 25)
}

emergency = {
    "Communication System": st.sidebar.selectbox("Communication System", QUAL_MAPPING)
}

data = {
    "Physical Security": physical,
    "Access Control": access,
    "Personnel": personnel,
    "Incident History": incident,
    "Emergency Preparedness": emergency
}

# ===============================
# COMPUTE SCORE
# ===============================
if st.button("📊 Compute Risk"):
    cat, contrib, overall = compute_scores(data)

    st.metric("Overall Risk Score", f"{overall}/100")

    badge = (
        "🟢 LOW" if overall <= 40 else
        "🟡 MODERATE" if overall <= 60 else
        "🟠 HIGH" if overall <= 80 else
        "🔴 CRITICAL"
    )
    st.subheader(badge)

# ===============================
# PREDICTIVE MODEL
# ===============================
if st.button("🤖 Run AI Risk Model"):
    model = joblib.load("security_multiorg_model.pkl")
    X = build_ml_features(data)

    expected = model.named_steps["preprocessor"].get_feature_names_out()
    X = X.reindex(columns=expected, fill_value=0)

    preds = model.predict_proba(X)

    st.session_state.model = model
    st.session_state.X = X
    st.session_state.preds = preds

    st.subheader("Predicted Risks")
    for i, label in enumerate(RISK_LABELS):
        st.metric(label, f"{preds[i][0][1]:.2%}")

# ===============================
# SHAP EXPLANATION
# ===============================
if "model" in st.session_state:
    st.subheader("🔍 Explain a Risk")
    risk = st.selectbox("Select Risk", RISK_LABELS)
    idx = RISK_LABELS.index(risk)

    clf = st.session_state.model.named_steps["clf"].estimators_[idx]
    shap_vals = get_shap_values(clf, st.session_state.X)

    fig, ax = plt.subplots()
    shap.bar_plot(shap_vals, feature_names=st.session_state.X.columns, show=False)
    st.pyplot(fig)

# ===============================
# WHAT-IF SIMULATION
# ===============================
st.subheader("🔁 What-If Simulation")
extra_guards = st.slider("Add Guards", 0, 20, 0)
improve_cctv = st.slider("Improve CCTV %", 0, 20, 0)

if "X" in st.session_state:
    X_sim = st.session_state.X.copy()
    X_sim["total_guards"] += extra_guards
    X_sim["cctv_functional_pct"] = np.clip(
        X_sim["cctv_functional_pct"] + improve_cctv, 0, 100
    )

    preds_sim = st.session_state.model.predict_proba(X_sim)

    for i, label in enumerate(RISK_LABELS):
        delta = preds_sim[i][0][1] - st.session_state.preds[i][0][1]
        st.metric(label, f"{preds_sim[i][0][1]:.2%}", f"{delta:+.2%}")

# ===============================
# ANOMALIES
# ===============================
st.subheader("🚨 Anomaly Detection")
if st.button("Detect Anomalies"):
    alerts = run_anomaly_engine(data)
    if not alerts:
        st.success("No anomalies detected")
    else:
        for a in alerts:
            st.error(f"{a[0]} anomaly ({a[1]}) | Z={a[2]}")
