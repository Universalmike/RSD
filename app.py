import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import joblib
import shap

# ================================
# CONFIG
# ================================
st.set_page_config(layout="wide")

# ================================
# QUALITATIVE MAPPING
# ================================
QUAL_MAPPING = {
    "Poor": 20,
    "Fair": 50,
    "Good": 80,
    "Excellent": 100
}

def map_score(v):
    return QUAL_MAPPING[v] if isinstance(v, str) else float(v)

# ================================
# SESSION STATE INIT
# ================================
for key in [
    "category_scores", "contributions", "overall",
    "show_dashboard", "model", "X_input", "preds"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ================================
# RISK SCORING ENGINE
# ================================
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
        avg = sum(scores) / len(scores)
        category_scores[cat] = round(avg, 2)
        contributions[cat] = round(avg * weights[cat], 2)

    overall = round(sum(contributions.values()), 2)
    return category_scores, contributions, overall

def risk_level(score):
    if score <= 40:
        return "🟢 LOW RISK"
    elif score <= 60:
        return "🟡 MODERATE RISK"
    elif score <= 80:
        return "🟠 HIGH RISK"
    return "🔴 CRITICAL RISK"

# ================================
# ML FEATURES
# ================================
def build_ml_features(data):
    return pd.DataFrame([{
        "size_employees": 580,
        "daily_visitors": 60,
        "facility_area_sqm": 22000,
        "cctv_coverage_pct": data["Physical Security"]["CCTV Coverage %"],
        "cctv_functional_pct": data["Physical Security"]["CCTV Functionality %"],
        "perimeter_cond": map_score(data["Physical Security"]["Perimeter Condition"]),
        "guard_score": data["Personnel"]["Guard Count Ratio Score"],
        "training_score": map_score(data["Personnel"]["Training Frequency"]),
        "background_check": map_score(data["Personnel"]["Background Checks"]),
        "incident_score": data["Incident History"]["Incident Severity Score"],
        "response_time": data["Incident History"]["Response Time Score"],
        "communication_score": map_score(data["Emergency Preparedness"]["Communication System"])
    }])

# ================================
# SHAP HELPERS
# ================================
def safe_shap_values(explainer, X):
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        return vals[1][0]
    return vals[0]

def save_shap_plot(values, names):
    fig, ax = plt.subplots(figsize=(6,4))
    shap.bar_plot(values, feature_names=names, max_display=8, show=False)
    plt.tight_layout()
    path = "shap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ================================
# ANOMALY ENGINE
# ================================
BASELINES = {
    "incident_score": (40, 10),
    "response_time": (60, 15),
    "cctv_functional_pct": (85, 5),
    "guard_score": (70, 10)
}

def run_anomaly_engine(X):
    alerts = []
    for col, (mean, std) in BASELINES.items():
        baseline = np.random.normal(mean, std, 30)
        z = (X[col].iloc[0] - baseline.mean()) / baseline.std()
        if abs(z) > 1.5:
            alerts.append((col, z))
    return alerts

# ================================
# PDF REPORT
# ================================
def generate_pdf(scores, contribs, overall, shap_img=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Security Risk Assessment Report", ln=True, align="C")
    pdf.ln(5)
    pdf.cell(0, 10, f"Overall Score: {overall}/100", ln=True)

    pdf.ln(5)
    for k, v in scores.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    if shap_img:
        pdf.add_page()
        pdf.image(shap_img, x=10, y=20, w=180)

    path = "security_report.pdf"
    pdf.output(path)
    return path

# ================================
# UI INPUTS
# ================================
st.title("🔐 Predictive Security Risk Dashboard")

st.sidebar.header("Facility Inputs")

physical = {
    "Perimeter Condition": st.sidebar.selectbox("Perimeter Condition", QUAL_MAPPING),
    "CCTV Coverage %": st.sidebar.number_input("CCTV Coverage", 0, 100, 35),
    "CCTV Functionality %": st.sidebar.number_input("CCTV Functionality", 0, 100, 55)
}

personnel = {
    "Guard Count Ratio Score": st.sidebar.number_input("Guard Adequacy", 0, 100, 40),
    "Training Frequency": st.sidebar.selectbox("Training", QUAL_MAPPING),
    "Background Checks": st.sidebar.selectbox("Background Checks", QUAL_MAPPING)
}

incidents = {
    "Incident Severity Score": st.sidebar.number_input("Incident Severity", 0, 100, 70),
    "Response Time Score": st.sidebar.number_input("Response Time", 0, 100, 45)
}

emergency = {
    "Communication System": st.sidebar.selectbox("Communication", QUAL_MAPPING)
}

data = {
    "Physical Security": physical,
    "Personnel": personnel,
    "Incident History": incidents,
    "Emergency Preparedness": emergency
}

# ================================
# ACTION BUTTONS
# ================================
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Compute Risk Score"):
        scores, contribs, overall = compute_scores(data)
        st.session_state.category_scores = scores
        st.session_state.contributions = contribs
        st.session_state.overall = overall
        st.session_state.show_dashboard = True

with col2:
    if st.button("🤖 Run Predictive Model"):
        st.session_state.model = joblib.load("security_multiorg_model.pkl")
        st.session_state.X_input = build_ml_features(data)
        st.session_state.preds = st.session_state.model.predict_proba(
            st.session_state.X_input
        )

# ================================
# DASHBOARD
# ================================
if st.session_state.show_dashboard:
    st.header("📊 Risk Dashboard")
    st.metric("Overall Risk Score", f"{st.session_state.overall}/100")
    st.markdown(f"### {risk_level(st.session_state.overall)}")

# ================================
# PREDICTION + SHAP
# ================================
if st.session_state.preds is not None:
    st.header("🤖 Predicted Threat Probabilities")

    labels = [
        "Unauthorized Access",
        "Insider Threat",
        "Emergency Failure",
        "Perimeter Breach"
    ]

    for i, lbl in enumerate(labels):
        st.metric(lbl, f"{st.session_state.preds[i][0][1]:.2%}")

    choice = st.selectbox("Explain Risk", labels)
    idx = labels.index(choice)

    est = st.session_state.model.named_steps["clf"].estimators_[idx]
    explainer = shap.TreeExplainer(est)
    shap_vals = safe_shap_values(explainer, st.session_state.X_input)

    st.subheader("🔍 SHAP Explanation")
    fig, ax = plt.subplots(figsize=(6,4))
    shap.bar_plot(shap_vals, feature_names=st.session_state.X_input.columns, show=False)
    st.pyplot(fig)

    shap_img = save_shap_plot(shap_vals, st.session_state.X_input.columns)
    pdf = generate_pdf(
        st.session_state.category_scores,
        st.session_state.contributions,
        st.session_state.overall,
        shap_img
    )
    with open(pdf, "rb") as f:
        st.download_button("📄 Download Report", f)

# ================================
# ANOMALY DETECTION
# ================================
st.header("🚨 Anomaly Detection")

if st.button("Detect Anomalies"):
    anomalies = run_anomaly_engine(st.session_state.X_input)
    if not anomalies:
        st.success("No anomalies detected")
    for col, z in anomalies:
        st.warning(f"{col} anomaly detected (Z={z:.2f})")

