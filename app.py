import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np
import json
from datetime import datetime
import sqlite3
import os

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Security Risk Assessment",
    page_icon="🔐",
    layout="wide"
)

# ----------------------------------
# DATABASE SETUP
# ----------------------------------

def init_database():
    """Initialize SQLite database with tables for facilities, assessments, and audit logs"""
    conn = sqlite3.connect('security_risk.db')
    c = conn.cursor()
    
    # Facilities table
    c.execute('''CREATE TABLE IF NOT EXISTS facilities
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  location TEXT,
                  industry TEXT,
                  size INTEGER,
                  created_date TEXT,
                  last_assessment TEXT)''')
    
    # Assessments table
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  facility_id INTEGER,
                  assessment_date TEXT,
                  overall_score REAL,
                  risk_level TEXT,
                  category_scores TEXT,
                  data_inputs TEXT,
                  recommendations TEXT,
                  FOREIGN KEY (facility_id) REFERENCES facilities(id))''')
    
    # Audit trail table
    c.execute('''CREATE TABLE IF NOT EXISTS audit_trail
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  facility_id INTEGER,
                  action TEXT,
                  timestamp TEXT,
                  user TEXT,
                  details TEXT,
                  FOREIGN KEY (facility_id) REFERENCES facilities(id))''')
    
    # Real-time monitoring table
    c.execute('''CREATE TABLE IF NOT EXISTS monitoring_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  facility_id INTEGER,
                  timestamp TEXT,
                  metric_type TEXT,
                  metric_value REAL,
                  alert_triggered INTEGER,
                  FOREIGN KEY (facility_id) REFERENCES facilities(id))''')
    
    # Budget table
    c.execute('''CREATE TABLE IF NOT EXISTS budget_allocations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  facility_id INTEGER,
                  category TEXT,
                  allocated_budget REAL,
                  spent_budget REAL,
                  fiscal_year INTEGER,
                  FOREIGN KEY (facility_id) REFERENCES facilities(id))''')
    
    conn.commit()
    conn.close()

def log_audit(facility_id, action, user, details):
    """Log actions to audit trail"""
    conn = sqlite3.connect('security_risk.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO audit_trail (facility_id, action, timestamp, user, details) VALUES (?, ?, ?, ?, ?)",
              (facility_id, action, timestamp, user, json.dumps(details)))
    conn.commit()
    conn.close()

def save_assessment(facility_id, overall_score, risk_level, category_scores, data_inputs, recommendations):
    """Save assessment to database"""
    conn = sqlite3.connect('security_risk.db')
    c = conn.cursor()
    assessment_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("""INSERT INTO assessments 
                 (facility_id, assessment_date, overall_score, risk_level, category_scores, data_inputs, recommendations)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (facility_id, assessment_date, overall_score, risk_level, 
               json.dumps(category_scores), json.dumps(data_inputs), json.dumps(recommendations)))
    
    # Update last assessment date in facilities
    c.execute("UPDATE facilities SET last_assessment = ? WHERE id = ?", (assessment_date, facility_id))
    
    conn.commit()
    conn.close()
    
    log_audit(facility_id, "assessment_completed", st.session_state.get('current_user', 'admin'),
              {"overall_score": overall_score, "risk_level": risk_level})

def get_facility_history(facility_id, limit=10):
    """Get assessment history for a facility"""
    conn = sqlite3.connect('security_risk.db')
    df = pd.read_sql_query(
        f"SELECT * FROM assessments WHERE facility_id = ? ORDER BY assessment_date DESC LIMIT ?",
        conn, params=(facility_id, limit))
    conn.close()
    return df

def get_all_facilities():
    """Get all facilities"""
    conn = sqlite3.connect('security_risk.db')
    df = pd.read_sql_query("SELECT * FROM facilities ORDER BY name", conn)
    conn.close()
    return df

def create_facility(name, location, industry, size):
    """Create new facility profile"""
    conn = sqlite3.connect('security_risk.db')
    c = conn.cursor()
    created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO facilities (name, location, industry, size, created_date) VALUES (?, ?, ?, ?, ?)",
              (name, location, industry, size, created_date))
    facility_id = c.lastrowid
    conn.commit()
    conn.close()
    
    log_audit(facility_id, "facility_created", st.session_state.get('current_user', 'admin'),
              {"name": name, "location": location})
    return facility_id

# ----------------------------------
# GEMINI AI INTEGRATION
# ----------------------------------

async def get_gemini_recommendations(data, category_scores, overall_score, api_key):
    """Get advanced recommendations from Gemini AI"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    prompt = f"""You are a security risk management expert. Analyze this facility security assessment and provide detailed, actionable recommendations.

FACILITY SECURITY DATA:
Overall Risk Score: {overall_score}/100 (lower is better)

Category Scores:
{json.dumps(category_scores, indent=2)}

Detailed Inputs:
{json.dumps(data, indent=2)}

Please provide:
1. Top 5 critical security gaps with specific remediation steps
2. Cost estimates (Low: <$5k, Medium: $5-20k, High: $20-50k, Very High: >$50k)
3. Implementation timeline (Immediate, 1-3 months, 3-6 months, 6-12 months)
4. Expected ROI and risk reduction percentage
5. Compliance implications (ISO 27001, NIST, etc.)
6. Quick wins (low-cost, high-impact improvements)
7. Long-term strategic recommendations

Format as JSON with this structure:
{{
  "critical_gaps": [
    {{
      "issue": "description",
      "severity": "Critical/High/Medium",
      "category": "Physical Security/Access Control/etc",
      "remediation": "specific steps",
      "cost": "Low/Medium/High/Very High",
      "timeline": "timeframe",
      "roi": "percentage",
      "risk_reduction": "percentage",
      "compliance": ["standards"]
    }}
  ],
  "quick_wins": [
    {{
      "action": "description",
      "impact": "description",
      "cost": "amount",
      "timeline": "timeframe"
    }}
  ],
  "strategic_initiatives": [
    {{
      "initiative": "description",
      "rationale": "why important",
      "investment": "amount",
      "timeline": "timeframe",
      "expected_outcome": "description"
    }}
  ],
  "executive_summary": "2-3 sentence overview of security posture and priorities"
}}"""

    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        text = response.text
        # Find JSON content
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
        return None
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return None

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

QUAL_MAPPING = {
    "Excellent": 10,
    "Good": 30,
    "Fair": 60,
    "Poor": 90
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
        "incident_score": f"Incident activity significantly higher than normal (Z={z:.2f})",
        "unauthorized_access_score": f"Unauthorized access attempts exceed historical patterns (Z={z:.2f})",
        "response_time_score": f"Security response time deviates from expected standards (Z={z:.2f})",
        "cctv_uptime": f"CCTV uptime below operational reliability levels (Z={z:.2f})",
        "guard_adequacy": f"Guard coverage insufficient compared to facility risk (Z={z:.2f})",
        "after_hours_security": f"After-hours security posture weaker than baseline (Z={z:.2f})"
    }
    return explanations.get(feature, "Unusual behavior detected")

def map_score(value):
    if isinstance(value, str):
        return QUAL_MAPPING[value]
    return 100 - float(value)

def map_score_direct(value):
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
        if cat == "Incident History":
            item_scores = [map_score_direct(v) for v in items.values()]
        else:
            item_scores = [map_score(v) for v in items.values()]
        avg_score = sum(item_scores) / len(item_scores)
        category_scores[cat] = round(avg_score, 2)
        contributions[cat] = round(avg_score * weights[cat], 2)
    overall_score = round(sum(contributions.values()), 2)
    return category_scores, contributions, overall_score

def build_anomaly_features(data):
    return {
        "incident_score": map_score_direct(data["Incident History"]["Incident Severity Score"]),
        "unauthorized_access_score": map_score_direct(data["Incident History"]["Incident Types Score"]),
        "response_time_score": map_score_direct(data["Incident History"]["Response Time Score"]),
        "cctv_uptime": 100 - map_score(data["Physical Security"]["CCTV Functionality %"]),
        "guard_adequacy": map_score(data["Personnel"]["Guard Count Ratio Score"]),
        "after_hours_security": map_score(data["Access Control"]["After-Hours Security"]),
    }

def risk_level(score):
    if score <= 30:
        return ("🟢 LOW RISK", "Low", "#28a745")
    elif score <= 50:
        return ("🟡 MODERATE RISK", "Moderate", "#ffc107")
    elif score <= 70:
        return ("🟠 HIGH RISK", "High", "#fd7e14")
    else:
        return ("🔴 CRITICAL RISK", "Critical", "#dc3545")

def save_chart_for_pdf(fig, filename):
    """Save matplotlib figure for PDF inclusion"""
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filename

def generate_enhanced_pdf(facility_name, category_scores, contributions, overall, 
                         recommendations, history_df=None, budget_data=None):
    """Generate comprehensive PDF report with visuals"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, txt="Security Risk Assessment Report", ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, txt=f"Facility: {facility_name}", ln=True, align='C')
    pdf.cell(0, 8, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    # Executive Summary
    badge, level, color = risk_level(overall)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Executive Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, txt=f"Overall Risk Score: {overall}/100 - {level}")
    pdf.ln(5)
    
    # Category Scores with visual bar
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, txt="Risk Breakdown by Category", ln=True)
    pdf.set_font("Arial", "", 10)
    
    for cat, score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
        pdf.cell(80, 6, txt=f"{cat}:", ln=False)
        pdf.cell(30, 6, txt=f"{score}/100", ln=False)
        
        # Simple text-based bar
        bar_length = int(score / 5)
        bar = "█" * bar_length
        pdf.cell(0, 6, txt=bar, ln=True)
    
    pdf.ln(5)
    
    # Add radar chart
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Risk Distribution Visualization", ln=True)
    
    labels = list(category_scores.keys())
    stats = list(category_scores.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats_plot = stats + stats[:1]
    angles_plot = angles + angles[:1]
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_plot, stats_plot, 'o-', linewidth=2, color='#dc3545')
    ax.fill(angles_plot, stats_plot, alpha=0.25, color='#dc3545')
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 100)
    ax.set_title("Risk Category Analysis", size=12, pad=20)
    ax.grid(True)
    
    radar_img = save_chart_for_pdf(fig, "radar_chart.png")
    pdf.image(radar_img, x=40, y=35, w=130)
    
    # Recommendations
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Priority Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)
    
    if recommendations and 'critical_gaps' in recommendations:
        for i, rec in enumerate(recommendations['critical_gaps'][:5], 1):
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, txt=f"{i}. [{rec.get('severity', 'High')}] {rec.get('issue', 'N/A')}")
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, txt=f"   Category: {rec.get('category', 'N/A')}")
            pdf.multi_cell(0, 5, txt=f"   Action: {rec.get('remediation', 'N/A')}")
            pdf.multi_cell(0, 5, txt=f"   Cost: {rec.get('cost', 'N/A')} | Timeline: {rec.get('timeline', 'N/A')}")
            pdf.multi_cell(0, 5, txt=f"   Expected Risk Reduction: {rec.get('risk_reduction', 'N/A')}")
            pdf.ln(3)
    
    # Historical Trend
    if history_df is not None and len(history_df) > 1:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Historical Risk Trend", ln=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = pd.to_datetime(history_df['assessment_date']).dt.strftime('%Y-%m-%d')
        scores = history_df['overall_score']
        ax.plot(dates, scores, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Assessment Date')
        ax.set_ylabel('Risk Score')
        ax.set_title('Risk Score Trend Over Time')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        trend_img = save_chart_for_pdf(fig, "trend_chart.png")
        pdf.image(trend_img, x=15, y=35, w=180)
    
    # Budget Allocation
    if budget_data:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Budget Allocation & Utilization", ln=True)
        
        categories = list(budget_data.keys())
        allocated = [budget_data[cat]['allocated'] for cat in categories]
        spent = [budget_data[cat]['spent'] for cat in categories]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, allocated, width, label='Allocated', color='#4CAF50')
        ax.bar(x + width/2, spent, width, label='Spent', color='#FF9800')
        ax.set_ylabel('Amount ($)')
        ax.set_title('Security Budget by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        
        budget_img = save_chart_for_pdf(fig, "budget_chart.png")
        pdf.image(budget_img, x=15, y=35, w=180)
    
    # Audit Trail Summary
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Report Metadata", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, txt=f"Generated by: {st.session_state.get('current_user', 'Admin')}", ln=True)
    pdf.cell(0, 6, txt=f"Report ID: RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}", ln=True)
    pdf.cell(0, 6, txt=f"System Version: 2.0", ln=True)
    
    file_path = f"security_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(file_path)
    
    # Cleanup temp images
    for img in ["radar_chart.png", "trend_chart.png", "budget_chart.png"]:
        if os.path.exists(img):
            os.remove(img)
    
    return file_path

def calculate_budget_roi(recommendations, current_budget):
    """Calculate ROI for security investments"""
    roi_analysis = []
    
    if recommendations and 'critical_gaps' in recommendations:
        for rec in recommendations['critical_gaps']:
            cost_map = {"Low": 3000, "Medium": 12500, "High": 35000, "Very High": 75000}
            cost = cost_map.get(rec.get('cost', 'Medium'), 12500)
            risk_reduction = float(rec.get('risk_reduction', '20').replace('%', ''))
            
            # Estimate annual loss prevented
            estimated_loss = 100000  # Base estimate
            annual_savings = estimated_loss * (risk_reduction / 100)
            roi = ((annual_savings - cost) / cost) * 100
            payback_months = (cost / annual_savings) * 12 if annual_savings > 0 else 999
            
            roi_analysis.append({
                "initiative": rec.get('issue', 'N/A'),
                "investment": cost,
                "annual_savings": annual_savings,
                "roi": round(roi, 1),
                "payback_months": round(payback_months, 1)
            })
    
    return roi_analysis

# Initialize database
init_database()

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "data_inputs" not in st.session_state:
    st.session_state.data_inputs = None
if "category_scores" not in st.session_state:
    st.session_state.category_scores = None
if "overall" not in st.session_state:
    st.session_state.overall = None
if "current_facility" not in st.session_state:
    st.session_state.current_facility = None
if "current_user" not in st.session_state:
    st.session_state.current_user = "admin"
if "gemini_recommendations" not in st.session_state:
    st.session_state.gemini_recommendations = None
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

# ----------------------------------
# MAIN UI
# ----------------------------------

st.title("🔐 Enterprise Security Risk Management Platform")
st.markdown("### Multi-Facility Security Assessment & Monitoring System")

# Facility Selection/Creation
st.sidebar.header("🏢 Facility Management")

facilities = get_all_facilities()

if len(facilities) == 0:
    st.sidebar.warning("No facilities found. Create one to get started.")
    with st.sidebar.expander("➕ Create New Facility", expanded=True):
        new_name = st.text_input("Facility Name")
        new_location = st.text_input("Location")
        new_industry = st.selectbox("Industry", ["Manufacturing", "Healthcare", "Education", "Retail", "Technology", "Finance", "Other"])
        new_size = st.number_input("Size (employees)", min_value=1, value=500)
        
        if st.button("Create Facility"):
            if new_name:
                facility_id = create_facility(new_name, new_location, new_industry, new_size)
                st.session_state.current_facility = facility_id
                st.success(f"✅ Facility '{new_name}' created!")
                st.rerun()
else:
    facility_options = {row['name']: row['id'] for _, row in facilities.iterrows()}
    selected_facility_name = st.sidebar.selectbox("Select Facility", list(facility_options.keys()))
    st.session_state.current_facility = facility_options[selected_facility_name]
    
    # Show facility info
    facility_info = facilities[facilities['id'] == st.session_state.current_facility].iloc[0]
    st.sidebar.info(f"""
    **Location:** {facility_info['location']}
    **Industry:** {facility_info['industry']}
    **Size:** {facility_info['size']} employees
    **Last Assessment:** {facility_info['last_assessment'] or 'Never'}
    """)
    
    with st.sidebar.expander("➕ Add New Facility"):
        new_name = st.text_input("Facility Name", key="new_fac")
        new_location = st.text_input("Location", key="new_loc")
        new_industry = st.selectbox("Industry", ["Manufacturing", "Healthcare", "Education", "Retail", "Technology", "Finance", "Other"], key="new_ind")
        new_size = st.number_input("Size (employees)", min_value=1, value=500, key="new_size")
        
        if st.button("Create Facility", key="create_btn"):
            if new_name:
                facility_id = create_facility(new_name, new_location, new_industry, new_size)
                st.success(f"✅ Facility '{new_name}' created!")
                st.rerun()

# Gemini API Key input
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your API key from Google AI Studio")
st.sidebar.caption("[Get API Key](https://makersuite.google.com/app/apikey)")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Assessment", 
    "📊 Dashboard", 
    "💰 Budget Optimizer",
    "📈 Executive View",
    "🔴 Live Monitoring",
    "📋 Audit Trail"
])

# TAB 1: ASSESSMENT
with tab1:
    if st.session_state.current_facility is None:
        st.warning("Please select or create a facility first.")
    else:
        st.header(f"Security Assessment - {selected_facility_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏢 Physical Security")
            physical = {
                "Perimeter Condition": st.selectbox("Perimeter Condition", ["Poor", "Fair", "Good", "Excellent"], key="peri", index=2),
                "CCTV Coverage %": st.number_input("CCTV Coverage (%)", 0, 100, 75, key="cctv_cov"),
                "CCTV Functionality %": st.number_input("Functional Cameras (%)", 0, 100, 85, key="cctv_func"),
                "Lighting Quality": st.selectbox("Lighting Quality", ["Poor", "Fair", "Good", "Excellent"], key="light", index=2),
                "Entry/Exit Control Quality": st.selectbox("Entry/Exit Control", ["Poor", "Fair", "Good", "Excellent"], key="entry", index=2)
            }
            
            st.subheader("🚪 Access Control")
            access = {
                "Visitor Management": st.selectbox("Visitor Management", ["Poor", "Fair", "Good", "Excellent"], key="visitor", index=2),
                "ID Verification": st.selectbox("ID Verification", ["Poor", "Fair", "Good", "Excellent"], key="id_ver", index=2),
                "Restricted Area Protection": st.selectbox("Restricted Area Protection", ["Poor", "Fair", "Good", "Excellent"], key="restrict", index=2),
                "After-Hours Security": st.selectbox("After-Hours Protocol", ["Poor", "Fair", "Good", "Excellent"], key="after_hours", index=2)
            }
            
            st.subheader("👮 Security Personnel")
            personnel = {
                "Guard Count Ratio Score": st.number_input("Guard Adequacy Score (0-100)", 0, 100, 30, key="guard_ratio"),
                "Training Frequency": st.selectbox("Training Frequency", ["Poor", "Fair", "Good", "Excellent"], key="training", index=2),
                "Background Checks": st.selectbox("Background Checks", ["Poor", "Fair", "Good", "Excellent"], key="bg_check", index=3),
                "Shift Coverage Quality": st.selectbox("Shift Coverage", ["Poor", "Fair", "Good", "Excellent"], key="shift", index=2)
            }
        
        with col2:
            st.subheader("📋 Incident History")
            incidents = {
                "Incident Severity Score": st.number_input("Incident Score (0-100)", 0, 100, 40, key="inc_sev"),
                "Incident Types Score": st.number_input("Incident Type Severity (0-100)", 0, 100, 35, key="inc_type"),
                "Response Time Score": st.number_input("Response Time Quality (0-100)", 0, 100, 60, key="resp_time"),
                "Documentation Quality": st.selectbox("Documentation Quality", ["Poor", "Fair", "Good", "Excellent"], key="doc_qual", index=2)
            }
            
            st.subheader("🚨 Emergency Preparedness")
            emergency = {
                "Emergency Plan": st.selectbox("Emergency Plan", ["Poor", "Fair", "Good", "Excellent"], key="emerg_plan", index=2),
                "Drill Frequency": st.selectbox("Drill Frequency", ["Poor", "Fair", "Good", "Excellent"], key="drill", index=1),
                "Communication System": st.selectbox("Communication System", ["Poor", "Fair", "Good", "Excellent"], key="comm", index=2),
                "Staff Readiness": st.selectbox("Staff Readiness", ["Poor", "Fair", "Good", "Excellent"], key="staff", index=2)
            }
        
        data = {
            "Physical Security": physical,
            "Access Control": access,
            "Personnel": personnel,
            "Incident History": incidents,
            "Emergency Preparedness": emergency
        }
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Run Standard Analysis", type="primary", use_container_width=True):
                category_scores, contributions, overall = compute_scores(data)
                badge, level, color = risk_level(overall)
                
                st.session_state.data_inputs = data
                st.session_state.category_scores = category_scores
                st.session_state.overall = overall
                
                save_assessment(st.session_state.current_facility, overall, level, 
                              category_scores, data, [])
                
                st.success("✅ Analysis complete!")
        
        with col2:
            if st.button("🤖 AI-Enhanced Analysis", type="primary", use_container_width=True):
                if not gemini_api_key:
                    st.error("Please enter your Gemini API key in the sidebar")
                else:
                    with st.spinner("Running AI analysis..."):
                        category_scores, contributions, overall = compute_scores(data)
                        badge, level, color = risk_level(overall)
                        
                        # Get Gemini recommendations
                        import asyncio
                        recommendations = asyncio.run(
                            get_gemini_recommendations(data, category_scores, overall, gemini_api_key)
                        )
                        
                        st.session_state.data_inputs = data
                        st.session_state.category_scores = category_scores
                        st.session_state.overall = overall
                        st.session_state.gemini_recommendations = recommendations
                        
                        save_assessment(st.session_state.current_facility, overall, level,
                                      category_scores, data, recommendations)
                        
                        st.success("✅ AI analysis complete!")
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.data_inputs = None
                st.session_state.category_scores = None
                st.session_state.overall = None
                st.rerun()

# TAB 2: DASHBOARD
with tab2:
    if st.session_state.overall is not None:
        st.header("📊 Risk Analysis Dashboard")
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        badge, level, color = risk_level(st.session_state.overall)
        
        with col1:
            st.metric("Overall Risk Score", f"{st.session_state.overall}/100", 
                     help="Lower is better")
        
        with col2:
            st.markdown(f"**Risk Level**")
            st.markdown(f"<h3 style='color: {color};'>{level}</h3>", unsafe_allow_html=True)
        
        with col3:
            # Get history for trend
            history = get_facility_history(st.session_state.current_facility, 2)
            if len(history) > 1:
                prev_score = history.iloc[1]['overall_score']
                delta = st.session_state.overall - prev_score
                st.metric("vs Previous", f"{delta:+.1f}", delta=f"{delta:+.1f}")
            else:
                st.metric("Assessments", len(history))
        
        with col4:
            if st.session_state.category_scores:
                worst = max(st.session_state.category_scores.items(), key=lambda x: x[1])
                st.metric("Top Risk Area", worst[0], f"{worst[1]}/100")
        
        st.markdown("---")
        
        # Visual Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Risk Scores")
            if st.session_state.category_scores:
                for cat, score in sorted(st.session_state.category_scores.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(f"**{cat}**: {score}/100")
                    st.progress(score / 100)
        
        with col2:
            st.subheader("Risk Distribution")
            labels = list(st.session_state.category_scores.keys())
            stats = list(st.session_state.category_scores.values())
            
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            stats_plot = stats + stats[:1]
            angles_plot = angles + angles[:1]
            
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(111, polar=True)
            ax.plot(angles_plot, stats_plot, 'o-', linewidth=2, color='#dc3545')
            ax.fill(angles_plot, stats_plot, alpha=0.25, color='#dc3545')
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, size=8)
            ax.set_ylim(0, 100)
            ax.grid(True)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # AI Recommendations
        if st.session_state.gemini_recommendations:
            st.subheader("🤖 AI-Generated Recommendations")
            
            recs = st.session_state.gemini_recommendations
            
            if 'executive_summary' in recs:
                st.info(f"**Executive Summary:** {recs['executive_summary']}")
            
            # Critical Gaps
            if 'critical_gaps' in recs:
                st.markdown("### 🚨 Critical Security Gaps")
                for i, gap in enumerate(recs['critical_gaps'][:5], 1):
                    with st.expander(f"#{i} [{gap.get('severity', 'High')}] {gap.get('issue', 'N/A')}", expanded=(i==1)):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Category:** {gap.get('category', 'N/A')}")
                            st.markdown(f"**Remediation:** {gap.get('remediation', 'N/A')}")
                            st.markdown(f"**Expected Impact:** {gap.get('risk_reduction', 'N/A')} risk reduction")
                            if gap.get('compliance'):
                                st.markdown(f"**Compliance:** {', '.join(gap['compliance'])}")
                        with col2:
                            st.markdown(f"**Cost:** {gap.get('cost', 'N/A')}")
                            st.markdown(f"**Timeline:** {gap.get('timeline', 'N/A')}")
                            st.markdown(f"**ROI:** {gap.get('roi', 'N/A')}")
            
            # Quick Wins
            if 'quick_wins' in recs:
                st.markdown("### ⚡ Quick Wins")
                cols = st.columns(3)
                for i, win in enumerate(recs['quick_wins'][:3]):
                    with cols[i]:
                        st.success(f"**{win.get('action', 'N/A')}**")
                        st.caption(f"Cost: {win.get('cost', 'N/A')}")
                        st.caption(f"Impact: {win.get('impact', 'N/A')}")
        
        # Download Report
        st.markdown("---")
        if st.button("📄 Generate Comprehensive PDF Report", use_container_width=True):
            history = get_facility_history(st.session_state.current_facility, 10)
            
            # Mock budget data (in real app, fetch from DB)
            budget_data = {
                "Physical Security": {"allocated": 50000, "spent": 35000},
                "Access Control": {"allocated": 40000, "spent": 38000},
                "Personnel": {"allocated": 120000, "spent": 115000},
                "Technology": {"allocated": 80000, "spent": 65000}
            }
            
            file_path = generate_enhanced_pdf(
                selected_facility_name,
                st.session_state.category_scores,
                {},
                st.session_state.overall,
                st.session_state.gemini_recommendations,
                history,
                budget_data
            )
            
            with open(file_path, "rb") as f:
                st.download_button("💾 Download PDF Report", f, 
                                 file_name=f"security_report_{selected_facility_name}.pdf",
                                 mime="application/pdf")
    
    else:
        st.info("Complete an assessment in the Assessment tab to view the dashboard.")

# TAB 3: BUDGET OPTIMIZER
with tab3:
    st.header("💰 Security Budget Optimizer")
    
    if st.session_state.gemini_recommendations and st.session_state.overall:
        st.subheader("Investment Analysis & ROI Calculator")
        
        # Current budget input
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.number_input("Total Annual Security Budget ($)", 
                                          min_value=0, value=300000, step=10000)
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", 
                                         ["Conservative", "Moderate", "Aggressive"])
        
        # Calculate ROI
        roi_analysis = calculate_budget_roi(st.session_state.gemini_recommendations, total_budget)
        
        if roi_analysis:
            st.subheader("📊 Investment Priorities (by ROI)")
            
            # Create DataFrame
            roi_df = pd.DataFrame(roi_analysis)
            roi_df = roi_df.sort_values('roi', ascending=False)
            
            # Display as table
            st.dataframe(roi_df, use_container_width=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # ROI chart
            ax1.barh(range(len(roi_df)), roi_df['roi'], color='#4CAF50')
            ax1.set_yticks(range(len(roi_df)))
            ax1.set_yticklabels([i[:30] + '...' if len(i) > 30 else i 
                                for i in roi_df['initiative']])
            ax1.set_xlabel('ROI (%)')
            ax1.set_title('Return on Investment')
            ax1.grid(axis='x', alpha=0.3)
            
            # Payback period
            ax2.barh(range(len(roi_df)), roi_df['payback_months'], color='#FF9800')
            ax2.set_yticks(range(len(roi_df)))
            ax2.set_yticklabels([i[:30] + '...' if len(i) > 30 else i 
                                for i in roi_df['initiative']])
            ax2.set_xlabel('Payback Period (months)')
            ax2.set_title('Investment Payback Timeline')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Budget allocation recommendation
            st.subheader("💡 Recommended Budget Allocation")
            
            total_investment = roi_df['investment'].sum()
            st.metric("Total Recommended Investment", f"${total_investment:,.0f}")
            
            if total_investment > total_budget:
                st.warning(f"⚠️ Recommended investments exceed budget by ${total_investment - total_budget:,.0f}")
                st.info("Consider prioritizing initiatives with highest ROI or increasing budget.")
            else:
                st.success(f"✅ Budget sufficient. Remaining: ${total_budget - total_investment:,.0f}")
    
    else:
        st.info("Run an AI-Enhanced Analysis first to access budget optimization features.")

# TAB 4: EXECUTIVE VIEW
with tab4:
    st.header("📈 Executive Dashboard")
    
    if st.session_state.current_facility:
        # Get historical data
        history = get_facility_history(st.session_state.current_facility, 12)
        
        if len(history) > 0:
            # Trend Analysis
            st.subheader("📉 Risk Trend Analysis")
            
            history['assessment_date'] = pd.to_datetime(history['assessment_date'])
            history = history.sort_values('assessment_date')
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history['assessment_date'], history['overall_score'], 
                   marker='o', linewidth=2, markersize=8, color='#1f77b4')
            ax.axhline(y=30, color='#28a745', linestyle='--', label='Low Risk Threshold')
            ax.axhline(y=50, color='#ffc107', linestyle='--', label='Moderate Risk Threshold')
            ax.axhline(y=70, color='#fd7e14', linestyle='--', label='High Risk Threshold')
            ax.set_xlabel('Assessment Date')
            ax.set_ylabel('Risk Score')
            ax.set_title('Security Risk Score Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                latest_score = history.iloc[-1]['overall_score']
                st.metric("Current Risk Score", f"{latest_score:.1f}/100")
            
            with col2:
                if len(history) > 1:
                    trend = history.iloc[-1]['overall_score'] - history.iloc[0]['overall_score']
                    st.metric("Overall Trend", f"{trend:+.1f}", delta=f"{trend:+.1f}")
                else:
                    st.metric("Total Assessments", len(history))
            
            with col3:
                avg_score = history['overall_score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}/100")
            
            with col4:
                best_score = history['overall_score'].min()
                st.metric("Best Score", f"{best_score:.1f}/100")
            
            # Risk Level Distribution
            st.subheader("🎯 Risk Level Distribution")
            
            def get_risk_category(score):
                if score <= 30:
                    return "Low"
                elif score <= 50:
                    return "Moderate"
                elif score <= 70:
                    return "High"
                else:
                    return "Critical"
            
            history['risk_category'] = history['overall_score'].apply(get_risk_category)
            risk_dist = history['risk_category'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}
            risk_colors = [colors.get(cat, '#999') for cat in risk_dist.index]
            ax.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%', 
                  colors=risk_colors, startangle=90)
            ax.set_title('Historical Risk Level Distribution')
            st.pyplot(fig)
            
        else:
            st.info("No historical data available yet. Complete an assessment to start tracking trends.")
    
    else:
        st.info("Select a facility to view executive dashboard.")

# TAB 5: LIVE MONITORING
with tab5:
    st.header("🔴 Real-Time Security Monitoring")
    
    if st.session_state.current_facility:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Live Metrics Dashboard")
        
        with col2:
            monitoring_toggle = st.toggle("Enable Monitoring", value=st.session_state.monitoring_active)
            st.session_state.monitoring_active = monitoring_toggle
        
        if st.session_state.monitoring_active:
            # Simulate real-time data
            st.info("🟢 Monitoring Active - Simulated data for demonstration")
            
            # Current status metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Simulate CCTV uptime
                cctv_uptime = np.random.uniform(92, 98)
                st.metric("CCTV Uptime", f"{cctv_uptime:.1f}%", 
                         delta=f"{np.random.uniform(-2, 2):.1f}%")
            
            with col2:
                active_guards = np.random.randint(8, 15)
                st.metric("Active Guards", active_guards)
            
            with col3:
                incidents_today = np.random.randint(0, 5)
                st.metric("Incidents Today", incidents_today)
            
            with col4:
                avg_response = np.random.uniform(5, 15)
                st.metric("Avg Response (min)", f"{avg_response:.1f}")
            
            # Live alerts
            st.subheader("🚨 Recent Alerts")
            
            # Simulate alerts
            alerts = [
                {"time": "14:23:15", "type": "INFO", "message": "Visitor check-in at Main Entrance"},
                {"time": "14:18:42", "type": "WARNING", "message": "Camera B-12 connection unstable"},
                {"time": "13:55:09", "type": "INFO", "message": "Shift change completed successfully"},
            ]
            
            for alert in alerts:
                if alert['type'] == "WARNING":
                    st.warning(f"**{alert['time']}** - {alert['message']}")
                else:
                    st.info(f"**{alert['time']}** - {alert['message']}")
            
            # Real-time chart (simulated)
            st.subheader("📊 Activity Trends (Last Hour)")
            
            times = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='5min')
            visitor_counts = np.random.poisson(8, 12)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(times, visitor_counts, marker='o', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Visitor Count')
            ax.set_title('Visitor Traffic (Last Hour)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("Enable monitoring to view real-time metrics.")
    
    else:
        st.info("Select a facility to enable monitoring.")

# TAB 6: AUDIT TRAIL
with tab6:
    st.header("📋 Audit Trail & Activity Log")
    
    if st.session_state.current_facility:
        # Fetch audit trail
        conn = sqlite3.connect('security_risk.db')
        audit_df = pd.read_sql_query(
            "SELECT * FROM audit_trail WHERE facility_id = ? ORDER BY timestamp DESC LIMIT 50",
            conn, params=(st.session_state.current_facility,))
        conn.close()
        
        if len(audit_df) > 0:
            st.subheader("Recent Activity")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                action_filter = st.multiselect("Filter by Action", 
                                              audit_df['action'].unique().tolist())
            with col2:
                user_filter = st.multiselect("Filter by User",
                                            audit_df['user'].unique().tolist())
            with col3:
                date_filter = st.date_input("Date Range", [])
            
            # Apply filters
            filtered_df = audit_df.copy()
            if action_filter:
                filtered_df = filtered_df[filtered_df['action'].isin(action_filter)]
            if user_filter:
                filtered_df = filtered_df[filtered_df['user'].isin(user_filter)]
            
            # Display audit log
            for _, row in filtered_df.iterrows():
                with st.expander(f"{row['timestamp']} - {row['action']} by {row['user']}"):
                    st.json(json.loads(row['details']))
            
            # Export option
            if st.button("📥 Export Audit Log (CSV)"):
                csv = filtered_df.to_csv(index=False)
                st.download_button("Download CSV", csv, 
                                 file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
                                 mime="text/csv")
        
        else:
            st.info("No audit trail records found for this facility yet.")
    
    else:
        st.info("Select a facility to view audit trail.")

# SIDEBAR STATS
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    if st.session_state.overall is not None:
        badge, level, color = risk_level(st.session_state.overall)
        st.markdown(f"**Current Risk:** {level}")
        st.markdown(f"**Score:** {st.session_state.overall}/100")
    
    if st.session_state.current_facility:
        history = get_facility_history(st.session_state.current_facility)
        st.markdown(f"**Total Assessments:** {len(history)}")
        
        conn = sqlite3.connect('security_risk.db')
        audit_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM audit_trail WHERE facility_id = ?",
            conn, params=(st.session_state.current_facility,)).iloc[0]['count']
        conn.close()
        st.markdown(f"**Audit Records:** {audit_count}")
    
    st.markdown("---")
    st.markdown("### 🔧 System Info")
    st.caption(f"Version: 2.0 Enterprise")
    st.caption(f"User: {st.session_state.current_user}")
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d')}")
