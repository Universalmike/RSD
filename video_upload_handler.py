"""
Safe Video Upload and Processing Handler for Streamlit
Handles all edge cases and errors properly
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

def process_uploaded_video(uploaded_file, monitor, max_frames=100, analyze_every_n=10):
    """
    Safely process uploaded video file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        monitor: CameraHealthMonitor instance
        max_frames: Maximum frames to analyze
        analyze_every_n: Analyze every Nth frame
    
    Returns:
        List of frame results or None if error
    """
    
    # Create temporary file with proper extension
    try:
        # Get file extension
        file_extension = Path(uploaded_file.name).suffix
        if not file_extension:
            file_extension = '.mp4'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Write uploaded content to temp file
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Verify file was created
        if not os.path.exists(temp_path):
            st.error("❌ Failed to create temporary file")
            return None
        
        # Try to open video
        cap = cv2.VideoCapture(temp_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            st.error(f"❌ Cannot open video file. The file might be corrupted or in an unsupported format.")
            st.info("💡 Try converting your video to MP4 (H.264) format")
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.success(f"✅ Video loaded successfully")
        st.info(f"📹 {width}x{height} | {total_frames} frames @ {fps:.1f} FPS")
        
        # Process video
        frame_results = []
        frame_count = 0
        analyzed_count = 0
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Display placeholders
        col1, col2 = st.columns(2)
        with col1:
            frame_display = st.empty()
        with col2:
            metrics_display = st.empty()
        
        while analyzed_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every Nth frame
            if frame_count % analyze_every_n == 0:
                try:
                    # Analyze frame
                    health_data = monitor.analyze_frame(frame)
                    
                    # Store results
                    frame_results.append({
                        "frame": frame_count,
                        "timestamp": frame_count / fps if fps > 0 else frame_count,
                        "health_score": health_data["health_score"],
                        "blur": health_data["metrics"]["blur_score"],
                        "brightness": health_data["metrics"]["brightness"],
                        "contrast": health_data["metrics"]["contrast"],
                        "noise": health_data["metrics"]["noise_level"],
                        "issues": len(health_data["issues"])
                    })
                    
                    analyzed_count += 1
                    
                    # Update progress
                    progress = min(analyzed_count / max_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing frame {frame_count}/{total_frames} ({analyzed_count}/{max_frames} analyzed)")
                    
                    # Update display every 10 analyzed frames
                    if analyzed_count % 10 == 0:
                        # Draw overlay
                        output = monitor.draw_overlay(frame, health_data)
                        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                        frame_display.image(output_rgb, use_column_width=True, 
                                          caption=f"Frame {frame_count}")
                        
                        # Show metrics
                        metrics_display.metric("Health Score", 
                                             f"{health_data['health_score']:.0f}/100",
                                             delta=None)
                
                except Exception as e:
                    st.warning(f"⚠️ Error processing frame {frame_count}: {str(e)}")
                    continue
        
        # Cleanup
        cap.release()
        
        try:
            os.unlink(temp_path)
        except:
            pass
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Processed {analyzed_count} frames")
        
        return frame_results
    
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        st.info("💡 Try a different video file or check the format")
        return None


def process_camera_photo(uploaded_photo, monitor):
    """
    Process single photo from camera or upload
    
    Args:
        uploaded_photo: Streamlit camera_input or file_uploader object
        monitor: CameraHealthMonitor instance
    
    Returns:
        Tuple of (frame, health_data) or (None, None) if error
    """
    try:
        # Open image
        image = Image.open(uploaded_photo)
        
        # Convert to OpenCV format
        frame = np.array(image)
        
        # Handle different color formats
        if len(frame.shape) == 2:
            # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Analyze
        health_data = monitor.analyze_frame(frame)
        
        return frame, health_data
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None


def display_health_results(frame, health_data, monitor):
    """
    Display health analysis results in a nice format
    
    Args:
        frame: OpenCV frame (BGR)
        health_data: Health analysis results
        monitor: CameraHealthMonitor instance
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Analyzed Image")
        # Draw overlay
        output = monitor.draw_overlay(frame, health_data)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.image(output_rgb, use_column_width=True)
    
    with col2:
        st.subheader("📊 Health Analysis")
        
        # Health score with color
        score = health_data["health_score"]
        if score >= 80:
            st.success(f"### Health Score: {score}/100")
            st.markdown("**Status:** ✅ Excellent")
        elif score >= 60:
            st.warning(f"### Health Score: {score}/100")
            st.markdown("**Status:** ⚠️ Good")
        elif score >= 40:
            st.warning(f"### Health Score: {score}/100")
            st.markdown("**Status:** ⚠️ Fair")
        else:
            st.error(f"### Health Score: {score}/100")
            st.markdown("**Status:** ❌ Poor")
        
        # Metrics
        st.markdown("#### Detailed Metrics")
        
        metrics_data = {
            "Blur Score": (health_data['metrics']['blur_score'], 
                          "Higher is sharper", 
                          health_data['metrics']['blur_score'] >= 100),
            "Brightness": (health_data['metrics']['brightness'], 
                          "50-200 is ideal",
                          50 <= health_data['metrics']['brightness'] <= 200),
            "Contrast": (health_data['metrics']['contrast'],
                        "Higher is better",
                        health_data['metrics']['contrast'] >= 30),
            "Noise Level": (health_data['metrics']['noise_level'],
                           "Lower is better",
                           health_data['metrics']['noise_level'] < 50)
        }
        
        for metric_name, (value, hint, is_good) in metrics_data.items():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.text(metric_name)
                st.caption(hint)
            with col_b:
                if is_good:
                    st.success(f"{value:.1f}")
                else:
                    st.error(f"{value:.1f}")
        
        # Issues
        st.markdown("#### Issues Detected")
        if health_data["issues"]:
            for issue in health_data["issues"]:
                issue_text = issue.replace('_', ' ').title()
                st.error(f"❌ {issue_text}")
        else:
            st.success("✅ No issues detected")


# =============================================================================
# EXAMPLE USAGE IN YOUR STREAMLIT APP
# =============================================================================

def camera_health_tab():
    """Complete camera health monitoring tab"""
    
    st.header("🎥 Camera Health Monitoring")
    
    # Import or initialize monitor
    try:
        from camera_health_monitor import CameraHealthMonitor
    except ImportError:
        st.error("❌ camera_health_monitor.py not found")
        st.info("Please make sure camera_health_monitor.py is in the same folder")
        return
    
    # Initialize monitor
    if "health_monitor" not in st.session_state:
        st.session_state.health_monitor = CameraHealthMonitor("CAM-001")
    
    monitor = st.session_state.health_monitor
    
    # Create tabs for different input methods
    input_method = st.radio(
        "Choose Input Method",
        ["📸 Camera Capture", "📁 Upload Video", "🖼️ Upload Images"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # =========================================================================
    # CAMERA CAPTURE
    # =========================================================================
    if input_method == "📸 Camera Capture":
        st.subheader("📸 Take a Photo")
        st.info("💡 Click 'Take photo' to capture from your webcam")
        
        camera_photo = st.camera_input("Camera")
        
        if camera_photo is not None:
            frame, health_data = process_camera_photo(camera_photo, monitor)
            
            if frame is not None and health_data is not None:
                display_health_results(frame, health_data, monitor)
    
    # =========================================================================
    # VIDEO UPLOAD
    # =========================================================================
    elif input_method == "📁 Upload Video":
        st.subheader("📁 Upload Video File")
        st.info("💡 Upload a video file to analyze camera health over time")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
            help="Supported formats: MP4, AVI, MOV, MKV, FLV, WMV"
        )
        
        if uploaded_video is not None:
            st.success(f"✅ Video uploaded: {uploaded_video.name} ({uploaded_video.size / (1024*1024):.1f} MB)")
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                analyze_every_n = st.slider(
                    "Analyze every N frames",
                    min_value=1,
                    max_value=30,
                    value=10,
                    help="Higher = faster but less detailed"
                )
            with col2:
                max_frames = st.slider(
                    "Maximum frames to analyze",
                    min_value=10,
                    max_value=500,
                    value=100,
                    help="Higher = more complete but slower"
                )
            
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                with st.spinner("Processing video..."):
                    results = process_uploaded_video(
                        uploaded_video,
                        monitor,
                        max_frames=max_frames,
                        analyze_every_n=analyze_every_n
                    )
                
                if results and len(results) > 0:
                    import pandas as pd
                    
                    st.success(f"✅ Analysis complete! Processed {len(results)} frames")
                    
                    # Convert to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Summary Statistics")
                    
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Blur Score")
                        st.line_chart(results_df.set_index("timestamp")["blur"])
                    with col2:
                        st.markdown("#### Brightness")
                        st.line_chart(results_df.set_index("timestamp")["brightness"])
                    
                    # Download results
                    st.markdown("---")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        f"video_analysis_{uploaded_video.name}.csv",
                        "text/csv",
                        key='download-video-csv'
                    )
    
    # =========================================================================
    # IMAGE UPLOAD
    # =========================================================================
    elif input_method == "🖼️ Upload Images":
        st.subheader("🖼️ Upload Images")
        st.info("💡 Upload one or more images to analyze")
        
        uploaded_files = st.file_uploader(
            "Choose images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded")
            
            if st.button("🔍 Analyze Images", type="primary", use_container_width=True):
                results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f"### Image {idx + 1}: {uploaded_file.name}")
                    
                    frame, health_data = process_camera_photo(uploaded_file, monitor)
                    
                    if frame is not None and health_data is not None:
                        display_health_results(frame, health_data, monitor)
                        
                        results.append({
                            "filename": uploaded_file.name,
                            "health_score": health_data["health_score"],
                            "blur": health_data["metrics"]["blur_score"],
                            "brightness": health_data["metrics"]["brightness"],
                            "issues": ", ".join(health_data["issues"]) if health_data["issues"] else "None"
                        })
                    
                    st.markdown("---")
                
                if results:
                    import pandas as pd
                    results_df = pd.DataFrame(results)
                    
                    st.subheader("📊 Batch Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "batch_analysis.csv",
                        "text/csv"
                    )


# To use in your main app.py:
if __name__ == "__main__":
    st.set_page_config(page_title="Camera Health Monitor", layout="wide")
    camera_health_tab()
