import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np
from risk_ass import *
from camera import CameraHealthMonitor
# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Risk-Security Diagnostic ",
    # page_icon="🔐",
    layout="wide"
)

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
# if "monitors" not in st.session_state:
#     st.session_state.monitors = {}
# if "camera_running" not in st.session_state:
#     st.session_state.camera_running = False
# if "health_history" not in st.session_state:
#     st.session_state.health_history = []


# ----------------------------------
# MAIN UI
# ----------------------------------

st.title("Risk-Security Diagnostic")
st.markdown("### Comprehensive facility security analysis and risk scoring")

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Data Input", "📊 Risk Analysis", "🤖 AI Predictions", "🔍 Anomaly Detection", "Camera Health"])

# ----------------------------------
# TAB 1: DATA INPUT
# ----------------------------------
with tab1:
    st.header("Facility Security Assessment")
    st.markdown("Complete all sections below to assess your facility's security posture.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Security")
        physical = {
            "Perimeter Condition": st.selectbox("Perimeter Condition", ["Poor", "Fair", "Good", "Excellent"], key="peri"),
            "CCTV Coverage %": st.number_input("CCTV Coverage (%)", 0, 100, 75, key="cctv_cov"),
            "CCTV Functionality %": st.number_input("Functional Cameras (%)", 0, 100, 85, key="cctv_func"),
            "Lighting Quality": st.selectbox("Lighting Quality", ["Poor", "Fair", "Good", "Excellent"], key="light"),
            "Entry/Exit Control Quality": st.selectbox("Entry/Exit Control", ["Poor", "Fair", "Good", "Excellent"], key="entry")
        }
        
        st.subheader("Access Control")
        access = {
            "Visitor Management": st.selectbox("Visitor Management", ["Poor", "Fair", "Good", "Excellent"], key="visitor"),
            "ID Verification": st.selectbox("ID Verification", ["Poor", "Fair", "Good", "Excellent"], key="id_ver"),
            "Restricted Area Protection": st.selectbox("Restricted Area Protection", ["Poor", "Fair", "Good", "Excellent"], key="restrict"),
            "After-Hours Security": st.selectbox("After-Hours Protocol", ["Poor", "Fair", "Good", "Excellent"], key="after_hours")
        }
        
        st.subheader("Security Personnel")
        personnel = {
            "Guard Count Ratio Score": st.number_input("Guard Adequacy Score (0-100)", 0, 100, 70, key="guard_ratio"),
            "Training Frequency": st.selectbox("Training Frequency", ["Poor", "Fair", "Good", "Excellent"], key="training"),
            "Background Checks": st.selectbox("Background Checks", ["Poor", "Fair", "Good", "Excellent"], key="bg_check"),
            "Shift Coverage Quality": st.selectbox("Shift Coverage", ["Poor", "Fair", "Good", "Excellent"], key="shift")
        }
    
    with col2:
        st.subheader("Incident History")
        incidents = {
            "Incident Severity Score": st.number_input("Incident Score (0-100)", 0, 100, 40, key="inc_sev"),
            "Incident Types Score": st.number_input("Incident Type Severity (0-100)", 0, 100, 35, key="inc_type"),
            "Response Time Score": st.number_input("Response Time Quality (0-100)", 0, 100, 60, key="resp_time"),
            "Documentation Quality": st.selectbox("Documentation Quality", ["Poor", "Fair", "Good", "Excellent"], key="doc_qual")
        }
        
        st.subheader("Emergency Preparedness")
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
                    st.download_button("Get PDF", pdf, file_name="security_report.pdf", use_container_width=True)
        
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
                    st.download_button("Get PDF Report", f, file_name="ai_security_report.pdf")
            
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

#TAB 5 - CAMERA HEALTH FEATURE

with tab5:
    st.header("⚙️ Configuration")
    
    camera_source = st.selectbox(
        "Camera Source",
        ["Webcam (0)", "Webcam (1)", "RTSP Stream", "Video File", "Simulated Feed"]
    )
    
    if camera_source == "RTSP Stream":
        rtsp_url = st.text_input("RTSP URL", "rtsp://admin:password@192.168.1.64:554/stream1")
        camera_id = st.text_input("Camera ID", "CAM-RTSP-001")
    elif camera_source == "Video File":
        video_file = st.text_input("Video File Path", "test_video.mp4")
        camera_id = st.text_input("Camera ID", "CAM-FILE-001")
    elif camera_source == "Simulated Feed":
        camera_id = st.text_input("Camera ID", "CAM-SIM-001")
    else:
        webcam_index = 0 if "0" in camera_source else 1
        camera_id = st.text_input("Camera ID", f"CAM-WEBCAM-{webcam_index}")
    
    st.markdown("---")
    
    st.subheader("🎯 Alert Thresholds")
    
    blur_threshold = st.slider("Blur Threshold", 50, 500, 100,
                               help="Below this value = blurry")
    brightness_min = st.slider("Min Brightness", 0, 100, 50)
    brightness_max = st.slider("Max Brightness", 150, 255, 200)
    fps_threshold = st.slider("Min FPS", 5, 30, 15)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
    with col2:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)


    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
        
    with col2:
        st.subheader("📊 Health Metrics")
        health_score_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
    
    # Issues and alerts section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Current Issues")
        issues_placeholder = st.empty()
    
    with col2:
        st.subheader("🔔 Recent Alerts")
        alerts_placeholder = st.empty()
    
    # Trends chart
    st.markdown("---")
    st.subheader("📈 Health Score Trend")
    chart_placeholder = st.empty()

    st.header("📁 Upload Video File")
    st.info("💡 Upload a video file (.mp4, .avi, .mov) to analyze")
    
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save uploaded file temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.success(f"✅ Video uploaded: {uploaded_video.name}")
        
        # Analysis settings
        col1, col2 = st.columns(2)
        with col1:
            analyze_every_n_frames = st.slider("Analyze every N frames", 1, 30, 10, 
                                              help="Process every Nth frame to speed up analysis")
        with col2:
            max_frames = st.slider("Maximum frames to analyze", 10, 300, 100)
        
        if st.button("🔍 Analyze Video", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.info(f"📹 Video Info: {total_frames} frames @ {fps} FPS")
            
            # Process video
            frame_results = []
            frame_count = 0
            analyzed_count = 0
            
            # Create placeholders for live updates
            col1, col2 = st.columns(2)
            with col1:
                current_frame_placeholder = st.empty()
            with col2:
                metrics_placeholder = st.empty()
            
            while analyzed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Analyze every Nth frame
                if frame_count % analyze_every_n_frames == 0:
                    health_data = st.session_state.monitor.analyze_frame(frame)
                    
                    frame_results.append({
                        "frame": frame_count,
                        "timestamp": frame_count / fps,
                        "health_score": health_data["health_score"],
                        "blur": health_data["metrics"]["blur_score"],
                        "brightness": health_data["metrics"]["brightness"],
                        "issues": len(health_data["issues"])
                    })
                    
                    analyzed_count += 1
                    
                    # Update progress
                    progress = analyzed_count / max_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing frame {frame_count}/{total_frames} ({analyzed_count}/{max_frames} analyzed)")
                    
                    # Show current frame every 10 analyzed frames
                    if analyzed_count % 10 == 0:
                        output = st.session_state.monitor.draw_overlay(frame, health_data)
                        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                        current_frame_placeholder.image(output_rgb, use_column_width=True, 
                                                       caption=f"Frame {frame_count}")
                        
                        # Show current metrics
                        metrics_placeholder.metric("Current Health Score", 
                                                   f"{health_data['health_score']:.0f}/100")
            
            cap.release()
            
            # Show results
            st.success(f"✅ Analysis complete! Processed {analyzed_count} frames")
            
            if frame_results:
                results_df = pd.DataFrame(frame_results)
                
                st.markdown("---")
                st.subheader("📊 Video Analysis Results")
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Health", f"{results_df['health_score'].mean():.1f}/100")
                with col2:
                    st.metric("Min Health", f"{results_df['health_score'].min():.1f}/100")
                with col3:
                    st.metric("Max Health", f"{results_df['health_score'].max():.1f}/100")
                with col4:
                    st.metric("Total Issues", int(results_df['issues'].sum()))
                
                # Charts
                st.markdown("#### Health Score Over Time")
                st.line_chart(results_df.set_index("timestamp")["health_score"])
                
                st.markdown("#### Detailed Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(results_df.set_index("timestamp")["blur"], 
                                 use_container_width=True)
                    st.caption("Blur Score (higher is sharper)")
                
                with col2:
                    st.line_chart(results_df.set_index("timestamp")["brightness"],
                                 use_container_width=True)
                    st.caption("Brightness (50-200 is ideal)")
                
                # Download results
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "video_analysis.csv",
                    "text/csv",
                    key='download-csv'
                )


# =============================================================================
# CAMERA FEED FUNCTIONS
# =============================================================================

def get_camera_source(source_type):
    """Get OpenCV VideoCapture object based on source type"""
    if "Webcam" in source_type:
        index = 0 if "(0)" in source_type else 1
        return cv2.VideoCapture(index)
    elif source_type == "RTSP Stream":
        return cv2.VideoCapture(rtsp_url)
    elif source_type == "Video File":
        return cv2.VideoCapture(video_file)
    elif source_type == "Simulated Feed":
        return SimulatedCamera()
    return None


class SimulatedCamera:
    """Simulate camera for testing without hardware"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.is_opened = True
    
    def isOpened(self):
        return self.is_opened
    
    def read(self):
        """Generate synthetic frame"""
        self.frame_count += 1
        
        # Create base frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some variation
        t = self.frame_count / 30  # time factor
        
        # Simulate different conditions
        if self.frame_count % 300 < 100:
            # Good quality
            brightness = 100 + 30 * np.sin(t)
            frame[:] = (brightness, brightness, brightness)
            
            # Add some pattern
            cv2.circle(frame, (320 + int(50*np.cos(t)), 240 + int(50*np.sin(t))), 
                      50, (255, 100, 0), -1)
            cv2.rectangle(frame, (100, 100), (540, 380), (0, 255, 255), 3)
            
        elif self.frame_count % 300 < 200:
            # Too dark
            frame[:] = (30, 30, 30)
            cv2.putText(frame, "DARK SCENE", (180, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
        else:
            # Blurry (add motion blur effect)
            frame[:] = (120, 120, 120)
            cv2.circle(frame, (320, 240), 80, (200, 200, 200), -1)
            
            # Apply blur
            frame = cv2.GaussianBlur(frame, (31, 31), 10)
            cv2.putText(frame, "BLURRY", (220, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Frame: {self.frame_count} | {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return True, frame
    
    def release(self):
        self.is_opened = False


def process_camera_stream():
    """Main processing loop"""
    # Initialize monitor
    monitor = CameraHealthMonitor(camera_id)
    
    # Update thresholds from sidebar
    monitor.thresholds["blur_min"] = blur_threshold
    monitor.thresholds["brightness_min"] = brightness_min
    monitor.thresholds["brightness_max"] = brightness_max
    monitor.thresholds["fps_min"] = fps_threshold
    
    # Get camera source
    cap = get_camera_source(camera_source)
    
    if not cap or not cap.isOpened():
        st.error("❌ Failed to open camera source")
        return
    
    st.session_state.camera_running = True
    frame_count = 0
    
    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("⚠️ Failed to read frame")
                break
            
            frame_count += 1
            
            # Analyze frame
            health_data = monitor.analyze_frame(frame)
            
            # Draw overlay
            output_frame = monitor.draw_overlay(frame, health_data)
            
            # Convert BGR to RGB for Streamlit
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            
            # Update display
            video_placeholder.image(output_frame, channels="RGB", use_column_width=True)
            
            # Update health score with color
            score = health_data["health_score"]
            if score >= 80:
                score_color = "green"
            elif score >= 60:
                score_color = "orange"
            else:
                score_color = "red"
            
            health_score_placeholder.markdown(
                f"<h1 style='text-align: center; color: {score_color};'>{score:.0f}/100</h1>",
                unsafe_allow_html=True
            )
            
            # Update status
            status = "🟢 ONLINE" if health_data["status"] == "online" else "🔴 OFFLINE"
            status_placeholder.markdown(f"### {status}")
            
            # Update metrics
            metrics = health_data["metrics"]
            metrics_df = pd.DataFrame({
                "Metric": ["Blur Score", "Brightness", "Contrast", "Noise", "FPS"],
                "Value": [
                    f"{metrics['blur_score']:.1f}",
                    f"{metrics['brightness']:.1f}",
                    f"{metrics['contrast']:.1f}",
                    f"{metrics['noise_level']:.1f}",
                    f"{metrics['fps']:.1f}"
                ]
            })
            metrics_placeholder.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Update issues
            issues = health_data.get("issues", [])
            if issues:
                issues_text = "\n".join([f"❌ **{issue.replace('_', ' ').title()}**" for issue in issues])
                issues_placeholder.markdown(issues_text)
            else:
                issues_placeholder.success("✅ No issues detected")
            
            # Update alerts
            alerts = health_data.get("alerts", [])
            if alerts:
                alerts_text = ""
                for alert in alerts[-5:]:
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢"
                    }
                    emoji = severity_emoji.get(alert["severity"], "ℹ️")
                    time_str = datetime.fromisoformat(alert["timestamp"]).strftime("%H:%M:%S")
                    alerts_text += f"{emoji} **{time_str}** - {alert['message']}\n\n"
                
                alerts_placeholder.markdown(alerts_text)
            else:
                alerts_placeholder.info("No recent alerts")
            
            # Store history for chart
            st.session_state.health_history.append({
                "timestamp": datetime.now(),
                "score": score
            })
            
            # Keep only last 100 points
            if len(st.session_state.health_history) > 100:
                st.session_state.health_history = st.session_state.health_history[-100:]
            
            # Update chart
            if len(st.session_state.health_history) > 1:
                history_df = pd.DataFrame(st.session_state.health_history)
                chart_placeholder.line_chart(
                    history_df.set_index("timestamp")["score"],
                    use_container_width=True
                )
            
            # Small delay to prevent overwhelming the UI
            time.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    finally:
        cap.release()
        st.session_state.camera_running = False


# =============================================================================
# BUTTON HANDLERS
# =============================================================================

    if start_btn:
        if not st.session_state.camera_running:
            with st.spinner("Starting camera..."):
                process_camera_stream()
    
    if stop_btn:
        st.session_state.camera_running = False
        st.success("Camera stopped")
    
    # Auto-refresh message
    if not st.session_state.camera_running:
        st.info("👆 Click **Start** to begin monitoring")
    else:
        st.info("Camera is running. Click **Stop** to end monitoring.")
    
    
    # Footer
    st.markdown("---")
    st.caption("Camera Health Monitoring System v1.0 | Phase 1 Implementation")
    

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
