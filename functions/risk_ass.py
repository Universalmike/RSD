# HELPER FUNCTIONS

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
