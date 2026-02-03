"""
Streamlit Camera Health Monitor - Web Compatible
Works properly in deployed Streamlit apps
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import io

# Import your health monitor
# Assuming you have camera_health_monitor.py in same folder
try:
    from camera import CameraHealthMonitor
except:
    st.error("⚠️ camera_health_monitor.py not found in the same folder")
    st.stop()

# Page config
st.set_page_config(
    page_title="Camera Health Monitor",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if "health_history" not in st.session_state:
    st.session_state.health_history = []
if "monitor" not in st.session_state:
    st.session_state.monitor = CameraHealthMonitor("CAM-001")
if "processing" not in st.session_state:
    st.session_state.processing = False

# Title
st.title("🎥 Camera Health Monitoring System")
st.markdown("### Real-time camera quality assessment")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📸 Live Camera Capture", "📁 Upload Video", "🎬 Upload Images"])

# =============================================================================
# TAB 1: LIVE CAMERA CAPTURE (Streamlit Native)
# =============================================================================

with tab1:
    st.header("📸 Live Camera Capture")
    st.info("💡 Click 'Take Photo' to capture from your webcam and analyze")
    
    # Streamlit's camera input
    camera_photo = st.camera_input("Take a photo")
    
    if camera_photo is not None:
        # Convert to OpenCV format
        image = Image.open(camera_photo)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process with health monitor
        health_data = st.session_state.monitor.analyze_frame(frame)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📹 Analyzed Image")
            # Draw overlay
            output = st.session_state.monitor.draw_overlay(frame, health_data)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, use_column_width=True)
        
        with col2:
            st.subheader("📊 Health Analysis")
            
            # Health score with color
            score = health_data["health_score"]
            if score >= 80:
                st.success(f"### Health Score: {score}/100")
            elif score >= 60:
                st.warning(f"### Health Score: {score}/100")
            else:
                st.error(f"### Health Score: {score}/100")
            
            # Metrics
            st.markdown("#### Detailed Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Blur Score", "Brightness", "Contrast", "Noise Level"],
                "Value": [
                    f"{health_data['metrics']['blur_score']:.1f}",
                    f"{health_data['metrics']['brightness']:.1f}",
                    f"{health_data['metrics']['contrast']:.1f}",
                    f"{health_data['metrics']['noise_level']:.1f}"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Issues
            if health_data["issues"]:
                st.markdown("#### ⚠️ Issues Detected")
                for issue in health_data["issues"]:
                    st.error(f"❌ {issue.replace('_', ' ').title()}")
            else:
                st.success("✅ No issues detected")
        
        # Store in history
        st.session_state.health_history.append({
            "timestamp": datetime.now(),
            "score": score
        })
        
        # Show trend if multiple captures
        if len(st.session_state.health_history) > 1:
            st.markdown("---")
            st.subheader("📈 Health Score History")
            history_df = pd.DataFrame(st.session_state.health_history)
            st.line_chart(history_df.set_index("timestamp")["score"])

# =============================================================================
# TAB 2: UPLOAD VIDEO FILE
# =============================================================================

with tab2:
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
# TAB 3: UPLOAD MULTIPLE IMAGES
# =============================================================================

with tab3:
    st.header("🎬 Upload Multiple Images")
    st.info("💡 Upload multiple images to batch analyze camera health")
    
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        if st.button("🔍 Analyze All Images", type="primary"):
            results = []
            
            # Create columns for display
            cols = st.columns(3)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Read image
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Analyze
                health_data = st.session_state.monitor.analyze_frame(frame)
                
                # Store results
                results.append({
                    "filename": uploaded_file.name,
                    "health_score": health_data["health_score"],
                    "blur": health_data["metrics"]["blur_score"],
                    "brightness": health_data["metrics"]["brightness"],
                    "issues": ", ".join(health_data["issues"]) if health_data["issues"] else "None"
                })
                
                # Display in grid
                col_idx = idx % 3
                with cols[col_idx]:
                    st.image(image, use_column_width=True, caption=uploaded_file.name)
                    
                    score = health_data["health_score"]
                    if score >= 80:
                        st.success(f"Health: {score:.0f}/100")
                    elif score >= 60:
                        st.warning(f"Health: {score:.0f}/100")
                    else:
                        st.error(f"Health: {score:.0f}/100")
            
            # Summary table
            st.markdown("---")
            st.subheader("📊 Batch Analysis Results")
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Health", f"{results_df['health_score'].mean():.1f}/100")
            with col2:
                st.metric("Best Image", results_df.loc[results_df['health_score'].idxmax(), 'filename'])
            with col3:
                st.metric("Worst Image", results_df.loc[results_df['health_score'].idxmin(), 'filename'])
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "batch_analysis.csv",
                "text/csv"
            )

# Sidebar info
with st.sidebar:
    st.header("ℹ️ How to Use")
    
    st.markdown("""
    ### 📸 Live Camera
    1. Click camera button
    2. Allow camera access in browser
    3. Take photo
    4. View instant analysis
    
    ### 📁 Video File
    1. Upload video (.mp4, .avi, etc)
    2. Adjust analysis settings
    3. Click "Analyze Video"
    4. Download results
    
    ### 🎬 Image Batch
    1. Upload multiple images
    2. Click "Analyze All"
    3. Compare results
    4. Download CSV report
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Health Score Guide")
    st.markdown("""
    - **80-100**: Excellent
    - **60-79**: Good
    - **40-59**: Fair
    - **0-39**: Poor
    """)
    
    if st.session_state.health_history:
        st.markdown("---")
        st.metric("Total Analyses", len(st.session_state.health_history))

# Footer
st.markdown("---")
st.caption("Camera Health Monitoring System v2.0 - Web Compatible")
