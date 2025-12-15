import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

# Map qualitative values to numeric scores
QUAL_MAPPING = {
    "Poor": 20,
    "Fair": 50,
    "Good": 80,
    "Excellent": 100
}

import numpy as np

def generate_baseline(feature_name, mean, std, n=30):
    return np.random.normal(mean, std, n)
    
BASELINES = {
    "incident_score": (40, 10),
    "unauthorized_access_score": (35, 10),
    "response_time_score": (60, 15),
    "cctv_uptime": (85, 5),
    "guard_adequacy": (70, 10),
    "after_hours_security": (65, 10)
}

def z_score_anomaly(current, baseline):
    mean = baseline.mean()
    std = baseline.std()
    if std == 0:
        return 0
    return (current - mean) / std

def run_anomaly_engine(data):
    features = build_anomaly_features(data)
    alerts = []

    for feature, value in features.items():
        mean, std = BASELINES[feature]
        baseline = generate_baseline(feature, mean, std)

        z = z_score_anomaly(value, baseline)

        if abs(z) >= 2:
            severity = "HIGH"
        elif abs(z) >= 1.3:
            severity = "MEDIUM"
        else:
            continue

        alerts.append({
            "feature": feature,
            "severity": severity,
            "z_score": round(z, 2),
            "value": value,
            "message": explain_anomaly(feature, value, z)
        })

    return alerts

def explain_anomaly(feature, value, z):
    explanations = {
        "incident_score": f"Incident activity is significantly higher than normal.",
        "unauthorized_access_score": "Unauthorized access attempts exceed historical patterns.",
        "response_time_score": "Security response time deviates from expected standards.",
        "cctv_uptime": "CCTV uptime has dropped below operational reliability levels.",
        "guard_adequacy": "Guard coverage is insufficient compared to facility risk.",
        "after_hours_security": "After-hours security posture is weaker than baseline."
    }

    return explanations.get(feature, "Unusual behavior detected.")

def map_score(value):
    if isinstance(value, str):
        return QUAL_MAPPING[value]
    return float(value)

def compute_scores(data):
    # Category weights
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
        item_scores = [map_score(v) for v in items.values()]
        avg_score = sum(item_scores) / len(item_scores)
        category_scores[cat] = round(avg_score, 2)
        contributions[cat] = round(avg_score * weights[cat], 2)

    overall_score = round(sum(contributions.values()), 2)

    return category_scores, contributions, overall_score

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



def generate_pdf(category_scores, contributions, overall, shap_img=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Security Risk Assessment Report", ln=True, align='C')
    pdf.ln(5)

    pdf.cell(200, 10, txt=f"Overall Security Score: {overall}/100", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(200, 10, txt="Category Scores", ln=True)

    pdf.set_font("Arial", size=10)
    for k, v in category_scores.items():
        pdf.cell(200, 7, txt=f"{k}: {v}", ln=True)

    if shap_img:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(200, 10, txt="AI Risk Explanation (SHAP)", ln=True)
        pdf.image(shap_img, x=15, y=30, w=180)

    file_path = "security_risk_report.pdf"
    pdf.output(file_path)
    return file_path


def get_shap_values(explainer, X, target_index=0):
    """
    Safely extract SHAP values for class 1 of a target
    """
    shap_vals = explainer.shap_values(X)

    # MultiOutput → pick target
    shap_target = shap_vals[target_index]

    # If binary class list, take class 1
    if isinstance(shap_target, list):
        return shap_target[1][0]

    # Else already (n_samples, n_features)
    return shap_target[0]

def save_shap_plot(shap_values, feature_names):
    fig, ax = plt.subplots(figsize=(6,4))
    shap.bar_plot(shap_values, feature_names=feature_names, max_display=8, show=False)
    plt.tight_layout()
    img_path = "shap_explanation.png"
    plt.savefig(img_path, dpi=150)
    plt.close()
    return img_path

def build_anomaly_features(data):
    return {
        "incident_score": data["Incident History"]["Incident Severity Score"],
        "unauthorized_access_score": data["Incident History"]["Incident Types Score"],
        "response_time_score": data["Incident History"]["Response Time Score"],
        "cctv_uptime": data["Physical Security"]["CCTV Functionality %"],
        "cctv_coverage": data["Physical Security"]["CCTV Coverage %"],
        "guard_adequacy": data["Personnel"]["Guard Count Ratio Score"],
        "after_hours_security": map_score(
            data["Access Control"]["After-Hours Security"]
        ),
    }



if "category_scores" not in st.session_state:
    st.session_state.category_scores = None
    st.session_state.contributions = None
    st.session_state.overall = None


# ----------------------------------
# STREAMLIT UI
# ----------------------------------

st.title("🔐 Security Risk Scoring Algorithm")
st.write("Modify the inputs below to compute the facility’s security risk score.")

st.header("1️⃣ Physical Security")
physical = {
    "Perimeter Condition": st.selectbox("Perimeter Condition", ["Poor", "Fair", "Good", "Excellent"]),
    "CCTV Coverage %": st.number_input("CCTV Coverage (%)", min_value=0, max_value=100, step=1),
    "CCTV Functionality %": st.number_input("Functional Cameras (%)", min_value=0, max_value=100, step=1),
    "Lighting Quality": st.selectbox("Lighting Quality", ["Poor", "Fair", "Good", "Excellent"]),
    "Entry/Exit Control Quality": st.selectbox("Entry/Exit Control", ["Poor", "Fair", "Good", "Excellent"])
}

st.header("2️⃣ Access Control")
access = {
    "Visitor Management": st.selectbox("Visitor Management", ["Poor", "Fair", "Good", "Excellent"]),
    "ID Verification": st.selectbox("ID Verification", ["Poor", "Fair", "Good", "Excellent"]),
    "Restricted Area Protection": st.selectbox("Restricted Area Protection", ["Poor", "Fair", "Good", "Excellent"]),
    "After-Hours Security": st.selectbox("After-Hours Protocol", ["Poor", "Fair", "Good", "Excellent"])
}

st.header("3️⃣ Security Personnel")
personnel = {
    "Guard Count Ratio Score": st.number_input("Guard Adequacy Score (0-100)", min_value=0, max_value=100),
    "Training Frequency": st.selectbox("Training Frequency", ["Poor", "Fair", "Good", "Excellent"]),
    "Background Checks": st.selectbox("Background Checks", ["Poor", "Fair", "Good", "Excellent"]),
    "Shift Coverage Quality": st.selectbox("Shift Coverage", ["Poor", "Fair", "Good", "Excellent"])
}

st.header("4️⃣ Incident History")
incidents = {
    "Incident Severity Score": st.number_input("Incident Score (0-100)", min_value=0, max_value=100),
    "Incident Types Score": st.number_input("Incident Type Severity (0-100)", min_value=0, max_value=100),
    "Response Time Score": st.number_input("Response Time Quality (0-100)", min_value=0, max_value=100),
    "Documentation Quality": st.selectbox("Documentation Quality", ["Poor", "Fair", "Good", "Excellent"])
}

st.header("5️⃣ Emergency Preparedness")
emergency = {
    "Emergency Plan": st.selectbox("Emergency Plan", ["Poor", "Fair", "Good", "Excellent"]),
    "Drill Frequency": st.selectbox("Drill Frequency", ["Poor", "Fair", "Good", "Excellent"]),
    "Communication System": st.selectbox("Communication System", ["Poor", "Fair", "Good", "Excellent"]),
    "Staff Readiness": st.selectbox("Staff Readiness", ["Poor", "Fair", "Good", "Excellent"])
}

data = {
    "Physical Security": physical,
    "Access Control": access,
    "Personnel": personnel,
    "Incident History": incidents,
    "Emergency Preparedness": emergency
}

st.markdown("---")
st.subheader("⚙️ Actions")

col1, col2 = st.columns(2)

with col1:
    compute_score_clicked = st.button("📊 Compute Risk Score")

with col2:
    predict_risk_clicked = st.button("🤖 Run Predictive Model")

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False
    
if st.session_state.show_dashboard:
    category_scores = st.session_state.category_scores
    contributions = st.session_state.contributions
    overall = st.session_state.overall

    # dashboard code here


if compute_score_clicked:
    category_scores, contributions, overall = compute_scores(data)

    st.session_state.category_scores = category_scores
    st.session_state.contributions = contributions
    st.session_state.overall = overall
    st.session_state.show_dashboard = True
       # ----------------------------
    # DASHBOARD SECTION
    # ----------------------------
    st.header("📊 Security Dashboard")

    st.metric(
    label="Security Risk Score",
    value=f"{overall}/100"
    )

    # ---- RISK LEVEL BADGE ----
    def risk_level(score):
        if score <= 40:
            return ("🟢 LOW RISK", "Low")
        elif score > 40 and score <= 60:
            return ("🟡 MODERATE RISK", "Moderate")
        elif score > 60 and score <=80:
            return ("🟠 HIGH RISK", "High")
        else:
            return ("🔴 CRITICAL RISK", "Critical")

    badge, level = risk_level(overall)
    st.subheader("Risk Rating")
    st.markdown(f"### {badge}")

    # ---- RADAR CHART ----
    st.subheader("Risk Breakdown (Radar Chart)")
    import numpy as np

    labels = list(category_scores.keys())
    stats = list(category_scores.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, stats)
    ax.fill(angles, stats, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # ---- TOP 3 WEAKNESSES ----
    st.subheader("🔻 Top 3 Weaknesses")
    weakest = sorted(category_scores.items(), key=lambda x: x[1])[:3]

    for area, score in weakest:
        st.write(f"**{area}:** {score}")

    # ---- AUTO RECOMMENDATIONS ----
    st.subheader("📌 Recommendations")

    recs = {
        "Physical Security": "Improve perimeter integrity, upgrade lighting, and increase CCTV coverage.",
        "Access Control": "Strengthen identity verification and improve restricted area policies.",
        "Personnel": "Increase guard training and ensure full shift coverage.",
        "Incident History": "Reduce incident frequency and improve response time & documentation.",
        "Emergency Preparedness": "Conduct regular drills and improve communication systems."
    }

    st.write("### Priority Recommendations")

    for area, score in weakest:
        st.write(f"**{area}:** {recs[area]}")


    # PDF report
    file_path = generate_pdf(category_scores, contributions, overall)
    with open(file_path, "rb") as pdf:
        st.download_button("📄 Download PDF Report", pdf, file_name="security_report.pdf")
# if "show_dashboard" not in st.session_state:
#     st.session_state.show_dashboard = False

if predict_risk_clicked:
    category_scores, contributions, overall = compute_scores(data)

    st.markdown("---")
    st.header("🤖 Predictive Security Risk Analysis")

    import joblib
    import shap

    # Load model (cache in real app)
    model = joblib.load("security_multiorg_model.pkl")

    X_input = build_ml_features(data)

    # Predict probabilities
    preds = model.predict_proba(X_input)

    risk_labels = [
        "Unauthorized Access",
        "Insider Threat",
        "Emergency Failure",
        "Perimeter Breach"
    ]

    st.subheader("📊 Predicted Risk Probabilities")

    risk_results = {}
    
    for i, label in enumerate(risk_labels):
        prob = preds[i][0][1]
        risk_results[label] = prob
        st.metric(label, f"{prob:.2%}")


        # ----------------------------
        # SHAP EXPLANATION
        # ----------------------------
    st.markdown("---")
    st.subheader("🔍 Explain a Specific Risk (SHAP)")
    
    selected_risk = st.selectbox(
        "Select risk to explain",
        risk_labels
    )

    target_index = risk_labels.index(selected_risk)

    
    rf_model = model.named_steps["clf"].estimators_[target_index]
    explainer = shap.TreeExplainer(rf_model)

    shap_values_safe = get_shap_values(
        explainer,
        X_input,
        target_index=0  # single estimator now
    )

        
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.bar_plot(
    shap_values_safe,
    feature_names=X_input.columns,
    max_display=8,
    show=False
    )
    st.pyplot(fig)
    
    shap_img = save_shap_plot(shap_values_safe, X_input.columns)

    file_path = generate_pdf(
    category_scores,
    contributions,
    overall,
    shap_img
    )

    with open(file_path, "rb") as f:
        st.download_button(
            "📄 Download AI Risk Report",
            f,
            file_name="security_risk_report.pdf"
        )

st.subheader("🔁 What-If Simulation")

extra_guards = st.slider("Add Extra Guards", 0, 20, 0)
improve_cctv = st.slider("Improve CCTV Functionality (%)", 0, 100, 0)

X_sim = X_input.copy()
X_sim["total_guards"] += extra_guards
X_sim["cctv_functional_pct"] = min(
    100, X_sim["cctv_functional_pct"].iloc[0] + improve_cctv
)
preds_sim = model.predict_proba(X_sim)

st.subheader("📉 Risk Change After Simulation")

for i, label in enumerate(risk_labels):
    before = preds[i][0][1]
    after = preds_sim[i][0][1]
    delta = after - before

    st.metric(
        label,
        f"{after:.2%}",
        delta=f"{delta:.2%}"
    )


st.markdown("---")
st.header("🚨 Security Anomaly Detection")

if st.button("🔍 Detect Anomalies"):
    anomalies = run_anomaly_engine(data)

    if not anomalies:
        st.success("✅ No significant anomalies detected.")
    else:
        for a in anomalies:
            if a["severity"] == "HIGH":
                st.error(f"🚨 {a['message']} (Z={a['z_score']})")
            else:
                st.warning(f"⚠ {a['message']} (Z={a['z_score']})")
