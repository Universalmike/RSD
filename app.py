import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

st.set_page_config(page_title="Security Risk Algorithm", layout="wide")

st.title("🔐 Security Risk Algorithm")


# ---------------------------------------------
#  SCORING FUNCTION
# ---------------------------------------------
def compute_score(data):
    weights = {
        "Physical Security": 0.25,
        "Access Control": 0.30,
        "Personnel": 0.15,
        "Incident History": 0.20,
        "Emergency Preparedness": 0.10
    }

    # Calculate category averages
    category_scores = {
        cat: sum(vals.values()) / len(vals)
        for cat, vals in data.items()
    }

    contributions = {
        k: category_scores[k] * weights[k]
        for k in category_scores
    }

    overall_score = sum(contributions.values())

    # Risk Level
    if overall_score < 30: level = "LOW"
    elif overall_score < 50: level = "MODERATE"
    elif overall_score < 70: level = "HIGH"
    elif overall_score < 85: level = "VERY HIGH"
    else: level = "CRITICAL"

    return category_scores, contributions, overall_score, level


# ---------------------------------------------
#   USER INPUT SECTION
# ---------------------------------------------
st.header("Enter Security Parameters")

with st.expander("A. Physical Security", expanded=True):
    physical = {
        "Perimeter Condition": st.slider("Perimeter Condition", 0, 100, 40),
        "CCTV Coverage": st.slider("CCTV Coverage (%)", 0, 100, 35),
        "Lighting Quality": st.slider("Lighting Quality", 0, 100, 45),
        "Entry/Exit Control": st.slider("Entry/Exit Control", 0, 100, 30),
    }

with st.expander("B. Access Control", expanded=False):
    access = {
        "Visitor Management": st.slider("Visitor Management", 0, 100, 20),
        "ID Verification": st.slider("ID Verification Level", 0, 100, 10),
        "Restricted Areas": st.slider("Restricted Area Protection", 0, 100, 5),
        "After Hours Protocols": st.slider("After Hours Protocols", 0, 100, 25),
    }

with st.expander("C. Personnel", expanded=False):
    personnel = {
        "Guard Count": st.slider("Guard Count Adequacy", 0, 100, 20),
        "Training Frequency": st.slider("Training Frequency", 0, 100, 10),
        "Background Checks": st.slider("Background Checks", 0, 100, 30),
        "Shift Coverage": st.slider("Shift Coverage", 0, 100, 40),
    }

with st.expander("D. Incident History", expanded=False):
    incidents = {
        "Incident Count Severity": st.slider("Incident Count Severity", 0, 100, 70),
        "Incident Types": st.slider("Severity of Incident Types", 0, 100, 60),
        "Response Time": st.slider("Average Response Time", 0, 100, 50),
        "Documentation Quality": st.slider("Documentation Quality", 0, 100, 20),
    }

with st.expander("E. Emergency Preparedness", expanded=False):
    emergency = {
        "Emergency Plan": st.slider("Emergency Plan Status", 0, 100, 5),
        "Drill Frequency": st.slider("Drill Frequency", 0, 100, 10),
        "Communication System": st.slider("Communication System", 0, 100, 5),
        "Staff Readiness": st.slider("Staff Readiness", 0, 100, 15),
    }


data = {
    "Physical Security": physical,
    "Access Control": access,
    "Personnel": personnel,
    "Incident History": incidents,
    "Emergency Preparedness": emergency
}


# ---------------------------------------------
# COMPUTE BUTTON
# ---------------------------------------------
if st.button("Compute Security Score"):
    cat_scores, contributions, overall, level = compute_score(data)

    st.subheader("📊 Risk Score Results")
    st.metric("Overall Security Score", f"{overall:.2f}")
    st.metric("Risk Level", level)

    st.write("### Category Scores")
    st.table(pd.DataFrame.from_dict(cat_scores, orient="index", columns=["Score"]))

    st.write("### Weighted Contributions")
    st.table(pd.DataFrame.from_dict(contributions, orient="index", columns=["Contribution"]))

    # Bar Chart
    fig, ax = plt.subplots()
    ax.bar(cat_scores.keys(), cat_scores.values())
    ax.set_title("Category Scores")
    st.pyplot(fig)

    # PDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Security Risk Assessment Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Overall Score: {overall:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {level}", ln=True)
    pdf.ln(10)

    for cat, val in cat_scores.items():
        pdf.cell(200, 8, txt=f"{cat}: {val:.2f}", ln=True)

    pdf_file = "security_report.pdf"
    pdf.output(pdf_file)

    st.download_button("Download PDF Report", data=open(pdf_file, "rb"), file_name="security_report.pdf")
