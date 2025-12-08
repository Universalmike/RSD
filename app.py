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


def generate_pdf(category_scores, contributions, overall):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Security Risk Assessment Report", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Overall Security Score: {overall}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(200, 10, txt="Category Scores:", ln=True)

    pdf.set_font("Arial", size=10)
    for k, v in category_scores.items():
        pdf.cell(200, 7, txt=f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(200, 10, txt="Weighted Contributions:", ln=True)

    pdf.set_font("Arial", size=10)
    for k, v in contributions.items():
        pdf.cell(200, 7, txt=f"{k}: {v}", ln=True)

    file_path = "security_risk_report.pdf"
    pdf.output(file_path)

    return file_path


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

if st.button("📊 Compute Security Risk Score"):
    category_scores, contributions, overall = compute_scores(data)

    st.subheader("📌 Results")
    st.write("### Overall Score:", overall)

    df_scores = pd.DataFrame({
        "Category": list(category_scores.keys()),
        "Score": list(category_scores.values())
    })

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(df_scores["Category"], df_scores["Score"])
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    st.pyplot(fig)

    # PDF report
    file_path = generate_pdf(category_scores, contributions, overall)
    with open(file_path, "rb") as pdf:
        st.download_button("📄 Download PDF Report", pdf, file_name="security_report.pdf")

