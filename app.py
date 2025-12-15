import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Risk-Security Diagnostic ",
    # page_icon="🔐",
    layout="wide"
)

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

QUAL_MAPPING = {
    "Poor": 90,
    "Fair": 60,
    "Good": 30,
    "Excellent": 10
}

BASELINES = {
    "incident_score": (40, 10),
    "unauthorized_access_score": (35, 10),
    "response_time_score": (60, 15),
    "cctv_uptime": (85, 5),
    "guard_adequacy": (70, 10),
    "after_hours_security": (65, 10)
}

def generate_baseline(feature_name, mean, std, n=30):
    return np.random.normal(mean, std, n)

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

def build_anomaly_features(data):
    return {
        "incident_score": data["Incident History"]["Incident Severity Score"],
        "unauthorized_access_score": data["Incident History"]["Incident Types Score"],
        "response_time_score": data["Incident History"]["Response Time Score"],
        "cctv_uptime": data["Physical Security"]["CCTV Functionality %"],
        "guard_adequacy": data["Personnel"]["Guard Count Ratio Score"],
        "after_hours_security": map_score(data["Access Control"]["After-Hours Security"]),
    }

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
    shap_vals = explainer.shap_values(X)
    shap_target = shap_vals[target_index]
    if isinstance(shap_target, list):
        return shap_target[1][0]
    return shap_target[0]

def save_shap_plot(shap_values, feature_names):
    import shap
    fig, ax = plt.subplots(figsize=(6,4))
    shap.bar_plot(shap_values, feature_names=feature_names, max_display=8, show=False)
    plt.tight_layout()
    img_path = "shap_explanation.png"
    plt.savefig(img_path, dpi=150)
    plt.close()
    return img_path

def risk_level(score):
    if score <= 40:
        return ("🟢 LOW RISK", "Low", "#28a745")
    elif score <= 60:
        return ("🟡 MODERATE RISK", "Moderate", "#ffc107")
    elif score <= 80:
        return ("🟠 HIGH RISK", "High", "#fd7e14")
    else:
        return ("🔴 CRITICAL RISK", "Critical", "#dc3545")

# ----------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------
if "data_inputs" not in st.session_state:
    st.session_state.data_inputs = None
if "category_scores" not in st.session_state:
    st.session_state.category_scores = None
if "contributions" not in st.session_state:
    st.session_state.contributions = None
if "overall" not in st.session_state:
    st.session_state.overall = None
if "X_input" not in st.session_state:
    st.session_state.X_input = None
if "ml_preds" not in st.session_state:
    st.session_state.ml_preds = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ----------------------------------
# MAIN UI
# ----------------------------------

st.title("Risk-Security Diagnostic")
st.markdown("### Comprehensive facility security analysis and risk scoring")

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📝 Data Input", "📊 Risk Analysis", "🤖 AI Predictions", "🔍 Anomaly Detection"])

# ----------------------------------
# TAB 1: DATA INPUT
# ----------------------------------
with tab1:
    st.header("Facility Security Assessment")
    st.markdown("Complete all sections below to assess your facility's security posture.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Physical Security")
        physical = {
            "Perimeter Condition": st.selectbox("Perimeter Condition", ["Poor", "Fair", "Good", "Excellent"], key="peri"),
            "CCTV Coverage %": st.number_input("CCTV Coverage (%)", 0, 100, 75, key="cctv_cov"),
            "CCTV Functionality %": st.number_input("Functional Cameras (%)", 0, 100, 85, key="cctv_func"),
            "Lighting Quality": st.selectbox("Lighting Quality", ["Poor", "Fair", "Good", "Excellent"], key="light"),
            "Entry/Exit Control Quality": st.selectbox("Entry/Exit Control", ["Poor", "Fair", "Good", "Excellent"], key="entry")
        }
        
        st.subheader("🚪 Access Control")
        access = {
            "Visitor Management": st.selectbox("Visitor Management", ["Poor", "Fair", "Good", "Excellent"], key="visitor"),
            "ID Verification": st.selectbox("ID Verification", ["Poor", "Fair", "Good", "Excellent"], key="id_ver"),
            "Restricted Area Protection": st.selectbox("Restricted Area Protection", ["Poor", "Fair", "Good", "Excellent"], key="restrict"),
            "After-Hours Security": st.selectbox("After-Hours Protocol", ["Poor", "Fair", "Good", "Excellent"], key="after_hours")
        }
        
        st.subheader("👮 Security Personnel")
        personnel = {
            "Guard Count Ratio Score": st.number_input("Guard Adequacy Score (0-100)", 0, 100, 70, key="guard_ratio"),
            "Training Frequency": st.selectbox("Training Frequency", ["Poor", "Fair", "Good", "Excellent"], key="training"),
            "Background Checks": st.selectbox("Background Checks", ["Poor", "Fair", "Good", "Excellent"], key="bg_check"),
            "Shift Coverage Quality": st.selectbox("Shift Coverage", ["Poor", "Fair", "Good", "Excellent"], key="shift")
        }
    
    with col2:
        st.subheader("📋 Incident History")
        incidents = {
            "Incident Severity Score": st.number_input("Incident Score (0-100)", 0, 100, 40, key="inc_sev"),
            "Incident Types Score": st.number_input("Incident Type Severity (0-100)", 0, 100, 35, key="inc_type"),
            "Response Time Score": st.number_input("Response Time Quality (0-100)", 0, 100, 60, key="resp_time"),
            "Documentation Quality": st.selectbox("Documentation Quality", ["Poor", "Fair", "Good", "Excellent"], key="doc_qual")
        }
        
        st.subheader("🚨 Emergency Preparedness")
        emergency = {
            "Emergency Plan": st.selectbox("Emergency Plan", ["Poor", "Fair", "Good", "Excellent"], key="emerg_plan"),
            "Drill Frequency": st.selectbox("Drill Frequency", ["Poor", "Fair", "Good", "Excellent"], key="drill"),
            "Communication System": st.selectbox("Communication System", ["Poor", "Fair", "Good", "Excellent"], key="comm"),
            "Staff Readiness": st.selectbox("Staff Readiness", ["Poor", "Fair", "Good", "Excellent"], key="staff")
        }
    
    data = {
        "Physical Security": physical,
        "Access Control": access,
        "Personnel": personnel,
        "Incident History": incidents,
        "Emergency Preparedness": emergency
    }
    
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔄 Reset All Inputs", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col_btn2:
        if st.button("📊 Analyze Risk Score", type="primary", use_container_width=True):
            category_scores, contributions, overall = compute_scores(data)
            st.session_state.data_inputs = data
            st.session_state.category_scores = category_scores
            st.session_state.contributions = contributions
            st.session_state.overall = overall
            st.session_state.analysis_complete = True
            st.success("✅ Analysis complete! Check the 'Risk Analysis' tab.")
    
    with col_btn3:
        if st.button("🤖 Run AI Model", type="primary", use_container_width=True):
            try:
                import joblib
                category_scores, contributions, overall = compute_scores(data)
                st.session_state.data_inputs = data
                st.session_state.category_scores = category_scores
                st.session_state.contributions = contributions
                st.session_state.overall = overall
                
                model = joblib.load("security_multiorg_model.pkl")
                X_input = build_ml_features(data)
                preds = model.predict_proba(X_input)
                
                st.session_state.X_input = X_input
                st.session_state.ml_preds = preds
                st.session_state.analysis_complete = True
                st.success("✅ AI analysis complete! Check the 'AI Predictions' tab.")
            except Exception as e:
                st.error(f"⚠️ Model file not found or error loading: {str(e)}")

# ----------------------------------
# TAB 2: RISK ANALYSIS
# ----------------------------------
with tab2:
    if st.session_state.analysis_complete and st.session_state.overall is not None:
        category_scores = st.session_state.category_scores
        contributions = st.session_state.contributions
        overall = st.session_state.overall
        
        # Overall Score Display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            badge, level, color = risk_level(overall)
            st.markdown(f"<h1 style='color: {color};'>{badge}</h1>", unsafe_allow_html=True)
            st.markdown(f"### Overall Security Score: **{overall}/100**")
        
        with col2:
            st.metric("Score", f"{overall}/100", delta=None)
        
        with col3:
            if st.button("📄 Download Report", use_container_width=True):
                file_path = generate_pdf(category_scores, contributions, overall)
                with open(file_path, "rb") as pdf:
                    st.download_button("💾 Get PDF", pdf, file_name="security_report.pdf", use_container_width=True)
        
        st.markdown("---")
        
        # Category Breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Category Scores")
            for cat, score in category_scores.items():
                st.progress(score / 100, text=f"{cat}: **{score}/100**")
        
        with col2:
            st.subheader("📈 Risk Distribution (Radar)")
            labels = list(category_scores.keys())
            stats = list(category_scores.values())
            
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            stats += stats[:1]
            angles += angles[:1]
            
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, stats, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, stats, alpha=0.25, color='#1f77b4')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=8)
            ax.set_ylim(0, 100)
            ax.grid(True)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Top Weaknesses & Recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔻 Top 3 Weaknesses")
            weakest = sorted(category_scores.items(), key=lambda x: x[1])[:3]
            for i, (area, score) in enumerate(weakest, 1):
                st.error(f"**{i}. {area}** - Score: {score}/100")
        
        with col2:
            st.subheader("📌 Priority Recommendations")
            recs = {
                "Physical Security": "Improve perimeter integrity, upgrade lighting, increase CCTV coverage.",
                "Access Control": "Strengthen identity verification and restricted area policies.",
                "Personnel": "Increase guard training and ensure full shift coverage.",
                "Incident History": "Reduce incident frequency, improve response time & documentation.",
                "Emergency Preparedness": "Conduct regular drills and improve communication systems."
            }
            
            for area, score in weakest:
                st.info(f"**{area}:** {recs[area]}")
    else:
        st.info("👈 Please complete the assessment in the 'Data Input' tab first.")

# ----------------------------------
# TAB 3: AI PREDICTIONS
# ----------------------------------
with tab3:
    if st.session_state.ml_preds is not None and st.session_state.X_input is not None:
        try:
            import joblib
            import shap
            
            preds = st.session_state.ml_preds
            X_input = st.session_state.X_input
            
            risk_labels = [
                "Unauthorized Access",
                "Insider Threat",
                "Emergency Failure",
                "Perimeter Breach"
            ]
            
            st.header("🤖 AI-Powered Risk Predictions")
            st.markdown("Machine learning predictions for specific security risks.")
            
            # Display predictions in columns
            cols = st.columns(4)
            for i, (label, col) in enumerate(zip(risk_labels, cols)):
                prob = preds[i][0][1]
                col.metric(label, f"{prob:.1%}", delta=None)
            
            st.markdown("---")
            
            # SHAP Explanation
            st.subheader("🔍 Explainable AI - Feature Impact Analysis")
            st.markdown("Select a risk to see which factors contribute most to the prediction.")
            
            selected_risk = st.selectbox("Select Risk Category", risk_labels, key="shap_select")
            target_index = risk_labels.index(selected_risk)
            
            model = joblib.load("security_multiorg_model.pkl")
            rf_model = model.named_steps["clf"].estimators_[target_index]
            explainer = shap.TreeExplainer(rf_model)
            
            shap_values_safe = get_shap_values(explainer, X_input, target_index=0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.bar_plot(shap_values_safe, feature_names=X_input.columns, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            
            if st.button("📄 Download AI Report", use_container_width=True):
                shap_img = save_shap_plot(shap_values_safe, X_input.columns)
                file_path = generate_pdf(
                    st.session_state.category_scores,
                    st.session_state.contributions,
                    st.session_state.overall,
                    shap_img
                )
                with open(file_path, "rb") as f:
                    st.download_button("💾 Get PDF Report", f, file_name="ai_security_report.pdf")
            
            st.markdown("---")
            
            # What-If Simulation
            st.subheader("🔮 What-If Scenario Simulator")
            st.markdown("Adjust security measures to see how risk predictions change.")
            
            col1, col2 = st.columns(2)
            with col1:
                extra_guards = st.slider("Additional Security Guards", 0, 20, 0, key="sim_guards")
            with col2:
                improve_cctv = st.slider("CCTV Functionality Improvement (%)", 0, 50, 0, key="sim_cctv")
            
            if extra_guards > 0 or improve_cctv > 0:
                X_sim = X_input.copy()
                X_sim["total_guards"] += extra_guards
                X_sim["cctv_functional_pct"] = min(100, X_sim["cctv_functional_pct"].iloc[0] + improve_cctv)
                preds_sim = model.predict_proba(X_sim)
                
                st.markdown("#### 📉 Risk Changes After Improvements")
                cols = st.columns(4)
                
                for i, (label, col) in enumerate(zip(risk_labels, cols)):
                    before = preds[i][0][1]
                    after = preds_sim[i][0][1]
                    delta = after - before
                    col.metric(label, f"{after:.1%}", delta=f"{delta:.1%}")
        
        except Exception as e:
            st.error(f"⚠️ Error in AI predictions: {str(e)}")
    else:
        st.info("👈 Please run the AI model from the 'Data Input' tab first.")

# ----------------------------------
# TAB 4: ANOMALY DETECTION
# ----------------------------------
with tab4:
    st.header("🚨 Security Anomaly Detection")
    st.markdown("Identify unusual patterns in your security metrics compared to baseline expectations.")
    
    if st.session_state.data_inputs is not None:
        if st.button("🔍 Run Anomaly Scan", type="primary", use_container_width=True):
            anomalies = run_anomaly_engine(st.session_state.data_inputs)
            
            if not anomalies:
                st.success("✅ No significant anomalies detected. All metrics within normal ranges.")
            else:
                st.warning(f"⚠️ Detected {len(anomalies)} anomalies requiring attention.")
                
                for a in anomalies:
                    if a["severity"] == "HIGH":
                        st.error(f"🚨 **HIGH SEVERITY** - {a['message']} (Z-Score: {a['z_score']})")
                    else:
                        st.warning(f"⚠️ **MEDIUM SEVERITY** - {a['message']} (Z-Score: {a['z_score']})")
    else:
        st.info("👈 Please complete the assessment in the 'Data Input' tab first.")

# ----------------------------------
# SIDEBAR
# ----------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Security+Risk+System", use_container_width=True)
    st.markdown("### 📊 Quick Stats")
    
    if st.session_state.overall is not None:
        badge, level, color = risk_level(st.session_state.overall)
        st.markdown(f"**Risk Level:** {level}")
        st.markdown(f"**Score:** {st.session_state.overall}/100")
    else:
        st.markdown("*No analysis completed yet*")
    
    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. **Input Data** - Enter your facility's security metrics
    2. **Analyze** - Get risk scores and recommendations
    3. **AI Predictions** - Machine learning risk forecasting
    4. **Detect Anomalies** - Identify unusual patterns
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("[📖 User Guide](#)")
    st.markdown("[💡 Best Practices](#)")
    st.markdown("[🔧 Support](#)")
