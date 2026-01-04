"""
Camera Health Monitoring System - Phase 1
Simple, focused implementation for monitoring camera feed quality
No ML required - uses computer vision fundamentals
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from collections import deque
import time

class CameraHealthMonitor:
    """
    Monitor camera health without machine learning
    Uses traditional CV techniques to assess video quality
    """
    
    def __init__(self, camera_id: str, history_size: int = 30):
        """
        Initialize health monitor
        
        Args:
            camera_id: Unique identifier for camera
            history_size: Number of recent frames to track
        """
        self.camera_id = camera_id
        self.history_size = history_size
        
        # Health metrics history
        self.blur_history = deque(maxlen=history_size)
        self.brightness_history = deque(maxlen=history_size)
        self.contrast_history = deque(maxlen=history_size)
        self.noise_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        
        # Frame tracking
        self.last_frame = None
        self.last_frame_time = None
        self.frame_count = 0
        self.total_frames = 0
        
        # Alert thresholds
        self.thresholds = {
            "blur_min": 100,           # Below this = blurry
            "brightness_min": 50,       # Below this = too dark
            "brightness_max": 200,      # Above this = overexposed
            "contrast_min": 30,         # Below this = low contrast
            "fps_min": 15,              # Below this = stuttering
            "obstruction_threshold": 0.8  # Similarity threshold
        }
        
        # Status
        self.is_online = False
        self.last_check_time = None
        self.alerts = []
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for health metrics
        
        Args:
            frame: BGR image from camera (numpy array)
            
        Returns:
            Dictionary with health metrics and scores
        """
        if frame is None or frame.size == 0:
            return self._create_offline_status()
        
        # Update status
        self.is_online = True
        self.last_check_time = datetime.now()
        self.total_frames += 1
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate all metrics
        blur_score = self._calculate_blur(gray)
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        noise_level = self._calculate_noise(gray)
        fps = self._calculate_fps()
        obstruction_score = self._detect_obstruction(gray)
        
        # Store in history
        self.blur_history.append(blur_score)
        self.brightness_history.append(brightness)
        self.contrast_history.append(contrast)
        self.noise_history.append(noise_level)
        self.fps_history.append(fps)
        
        # Detect issues
        issues = self._detect_issues(blur_score, brightness, contrast, noise_level, fps, obstruction_score)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(blur_score, brightness, contrast, noise_level, fps)
        
        # Create alerts for issues
        self._generate_alerts(issues)
        
        # Store current frame for next comparison
        self.last_frame = gray.copy()
        
        return {
            "camera_id": self.camera_id,
            "timestamp": datetime.now().isoformat(),
            "status": "online",
            "health_score": round(health_score, 2),
            "metrics": {
                "blur_score": round(blur_score, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "noise_level": round(noise_level, 2),
                "fps": round(fps, 2),
                "obstruction_score": round(obstruction_score, 2)
            },
            "issues": issues,
            "alerts": self.alerts[-5:],  # Last 5 alerts
            "frame_count": self.total_frames
        }
    
    def _calculate_blur(self, gray: np.ndarray) -> float:
        """
        Calculate image blur using Laplacian variance
        Higher values = sharper image
        Lower values = more blur
        
        Typical ranges:
        - > 500: Very sharp
        - 100-500: Acceptable
        - < 100: Blurry
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """
        Calculate average brightness (0-255)
        
        Typical ranges:
        - < 50: Too dark
        - 50-200: Good
        - > 200: Overexposed
        """
        return np.mean(gray)
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation
        Higher values = more contrast
        
        Typical ranges:
        - > 50: Good contrast
        - 30-50: Acceptable
        - < 30: Low contrast (washed out)
        """
        return np.std(gray)
    
    def _calculate_noise(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using high-frequency content
        Higher values = more noise
        
        Typical ranges:
        - < 20: Clean image
        - 20-50: Acceptable
        - > 50: Noisy
        """
        # Use a high-pass filter to detect noise
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        return np.std(filtered)
    
    def _calculate_fps(self) -> float:
        """
        Calculate actual frames per second
        """
        current_time = time.time()
        
        if self.last_frame_time is None:
            self.last_frame_time = current_time
            return 0.0
        
        time_diff = current_time - self.last_frame_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        
        self.last_frame_time = current_time
        return fps
    
    def _detect_obstruction(self, gray: np.ndarray) -> float:
        """
        Detect if lens is obstructed/covered
        
        Returns:
            Similarity score (0-1)
            - Close to 1.0 = likely obstructed (uniform image)
            - Close to 0.0 = normal (varied content)
        """
        if self.last_frame is None:
            return 0.0
        
        # Compare with previous frame
        # If frames are too similar, might indicate obstruction
        diff = cv2.absdiff(gray, self.last_frame)
        mean_diff = np.mean(diff)
        
        # Also check for uniform color (solid obstruction)
        std_dev = np.std(gray)
        
        # Low variance + low frame difference = possible obstruction
        if std_dev < 10:  # Very uniform image
            return 1.0
        elif mean_diff < 5:  # Very little change
            return 0.8
        else:
            return 0.0
    
    def _detect_issues(self, blur: float, brightness: float, contrast: float, 
                      noise: float, fps: float, obstruction: float) -> List[str]:
        """
        Detect specific issues based on thresholds
        """
        issues = []
        
        if blur < self.thresholds["blur_min"]:
            issues.append("blurry_image")
        
        if brightness < self.thresholds["brightness_min"]:
            issues.append("too_dark")
        elif brightness > self.thresholds["brightness_max"]:
            issues.append("overexposed")
        
        if contrast < self.thresholds["contrast_min"]:
            issues.append("low_contrast")
        
        if noise > 50:
            issues.append("high_noise")
        
        if fps < self.thresholds["fps_min"] and fps > 0:
            issues.append("low_fps")
        
        if obstruction > self.thresholds["obstruction_threshold"]:
            issues.append("lens_obstruction")
        
        return issues
    
    def _calculate_health_score(self, blur: float, brightness: float, 
                                contrast: float, noise: float, fps: float) -> float:
        """
        Calculate overall health score (0-100)
        100 = perfect health
        0 = severe issues
        """
        score = 100.0
        
        # Blur penalty (0-30 points)
        if blur < 50:
            score -= 30
        elif blur < 100:
            score -= 20
        elif blur < 200:
            score -= 10
        
        # Brightness penalty (0-20 points)
        if brightness < 40 or brightness > 220:
            score -= 20
        elif brightness < 50 or brightness > 200:
            score -= 10
        
        # Contrast penalty (0-20 points)
        if contrast < 20:
            score -= 20
        elif contrast < 30:
            score -= 10
        
        # Noise penalty (0-15 points)
        if noise > 60:
            score -= 15
        elif noise > 50:
            score -= 10
        
        # FPS penalty (0-15 points)
        if fps > 0 and fps < 10:
            score -= 15
        elif fps > 0 and fps < 15:
            score -= 10
        
        return max(0, score)
    
    def _generate_alerts(self, issues: List[str]):
        """Generate alerts for detected issues"""
        severity_map = {
            "blurry_image": "medium",
            "too_dark": "high",
            "overexposed": "medium",
            "low_contrast": "low",
            "high_noise": "low",
            "low_fps": "medium",
            "lens_obstruction": "critical"
        }
        
        for issue in issues:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "issue": issue,
                "severity": severity_map.get(issue, "low"),
                "message": self._get_issue_message(issue)
            }
            self.alerts.append(alert)
    
    def _get_issue_message(self, issue: str) -> str:
        """Get human-readable message for issue"""
        messages = {
            "blurry_image": "Camera image is out of focus or blurry",
            "too_dark": "Camera image is too dark - check lighting or camera settings",
            "overexposed": "Camera image is overexposed - reduce exposure or add shade",
            "low_contrast": "Image has low contrast - may be foggy or need cleaning",
            "high_noise": "High noise detected - possible low light or sensor issue",
            "low_fps": "Frame rate is below normal - check network or camera load",
            "lens_obstruction": "Lens may be obstructed, dirty, or covered"
        }
        return messages.get(issue, "Unknown issue detected")
    
    def _create_offline_status(self) -> Dict:
        """Create status for offline camera"""
        return {
            "camera_id": self.camera_id,
            "timestamp": datetime.now().isoformat(),
            "status": "offline",
            "health_score": 0,
            "metrics": {},
            "issues": ["offline"],
            "alerts": [],
            "frame_count": self.total_frames
        }
    
    def get_trends(self) -> Dict:
        """
        Get trending metrics over recent history
        Useful for detecting gradual degradation
        """
        if not self.blur_history:
            return {}
        
        return {
            "blur_trend": {
                "current": self.blur_history[-1] if self.blur_history else 0,
                "average": np.mean(self.blur_history),
                "min": np.min(self.blur_history),
                "max": np.max(self.blur_history)
            },
            "brightness_trend": {
                "current": self.brightness_history[-1] if self.brightness_history else 0,
                "average": np.mean(self.brightness_history),
                "min": np.min(self.brightness_history),
                "max": np.max(self.brightness_history)
            },
            "fps_trend": {
                "current": self.fps_history[-1] if self.fps_history else 0,
                "average": np.mean(self.fps_history),
                "min": np.min(self.fps_history),
                "max": np.max(self.fps_history)
            }
        }
    
    def draw_overlay(self, frame: np.ndarray, health_data: Dict) -> np.ndarray:
        """
        Draw health metrics overlay on frame
        
        Args:
            frame: Original camera frame
            health_data: Health data from analyze_frame()
            
        Returns:
            Frame with overlay drawn
        """
        output = frame.copy()
        height, width = output.shape[:2]
        
        # Determine health color
        score = health_data["health_score"]
        if score >= 80:
            color = (0, 255, 0)  # Green
        elif score >= 60:
            color = (0, 255, 255)  # Yellow
        elif score >= 40:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Draw semi-transparent overlay bar at top
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # Camera ID and timestamp
        cv2.putText(output, f"Camera: {self.camera_id}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(output, timestamp, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Health score
        cv2.putText(output, f"Health: {score:.0f}%", (width - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS
        fps = health_data["metrics"].get("fps", 0)
        cv2.putText(output, f"FPS: {fps:.1f}", (width - 180, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Issues (if any)
        issues = health_data.get("issues", [])
        if issues:
            y_offset = 100
            cv2.putText(output, "ISSUES:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for issue in issues[:3]:  # Show max 3 issues
                y_offset += 25
                issue_text = issue.replace("_", " ").upper()
                cv2.putText(output, f"• {issue_text}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return output


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def test_with_webcam():
    """Test camera health monitor with webcam"""
    print("🎥 Starting Camera Health Monitor Test...")
    print("Press 'q' to quit\n")
    
    # Initialize monitor
    monitor = CameraHealthMonitor("WEBCAM-001")
    
    # Open webcam (or use video file: cv2.VideoCapture("video.mp4"))
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    print("Analyzing frames...\n")
    
    frame_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Analyze frame
            health_data = monitor.analyze_frame(frame)
            
            # Draw overlay
            output = monitor.draw_overlay(frame, health_data)
            
            # Display
            cv2.imshow("Camera Health Monitor", output)
            
            # Print status every 30 frames
            frame_counter += 1
            if frame_counter % 30 == 0:
                print(f"\n📊 Frame {frame_counter} Status:")
                print(f"   Health Score: {health_data['health_score']}/100")
                print(f"   FPS: {health_data['metrics']['fps']:.1f}")
                print(f"   Blur Score: {health_data['metrics']['blur_score']:.1f}")
                print(f"   Brightness: {health_data['metrics']['brightness']:.1f}")
                
                if health_data['issues']:
                    print(f"   ⚠️  Issues: {', '.join(health_data['issues'])}")
                else:
                    print(f"   ✅ No issues detected")
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final summary
        print("\n" + "="*50)
        print("📈 SESSION SUMMARY")
        print("="*50)
        
        trends = monitor.get_trends()
        if trends:
            print(f"\nBlur Score: {trends['blur_trend']['average']:.1f} (avg)")
            print(f"Brightness: {trends['brightness_trend']['average']:.1f} (avg)")
            print(f"FPS: {trends['fps_trend']['average']:.1f} (avg)")
        
        print(f"\nTotal Frames Processed: {monitor.total_frames}")
        print(f"Total Alerts: {len(monitor.alerts)}")
        
        if monitor.alerts:
            print("\n⚠️  Recent Alerts:")
            for alert in monitor.alerts[-5:]:
                print(f"   [{alert['severity'].upper()}] {alert['message']}")


def test_with_rtsp():
    """Test with RTSP camera stream"""
    # Replace with your camera's RTSP URL
    rtsp_url = "rtsp://admin:password@192.168.1.64:554/stream1"
    
    monitor = CameraHealthMonitor("RTSP-CAM-001")
    cap = cv2.VideoCapture(rtsp_url)
    
    # Same logic as webcam test...
    # (Implementation same as test_with_webcam)


if __name__ == "__main__":
    # Run webcam test
    test_with_webcam()
    
    # Or test with RTSP camera:
    # test_with_rtsp()
