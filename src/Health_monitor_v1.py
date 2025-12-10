"""
Enhanced Posture Detection with Gradio Interface
Features:
- Real-time video monitoring with overlay
- Session-based tracking with daily statistics
- Interactive Gradio dashboard
- Water intake monitoring (hand + head tilt detection)
- Posture analysis
- Screen time tracking
"""

import cv2
import time
import math as m
import mediapipe as mp
import numpy as np
import gradio as gr
import argparse
import os
from datetime import datetime
import threading
import queue
from collections import defaultdict
import json

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def findDistance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    dist = m.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    """Calculate angle subtended by line to y-axis."""
    try:
        theta = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
        degree = int(180/m.pi) * theta
        return degree
    except:
        return 0

def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
# WATER INTAKE TRACKER CLASS
# ============================================================================

class WaterIntakeTracker:
    """Track water drinking behavior using hand-to-mouth and head tilt detection."""

    def __init__(self):
        self.drinking_frames = 0
        self.drinking_threshold = 20  # 0.67 seconds at 30 FPS (reduced for faster detection)
        self.drinking_event_active = False
        self.cooldown_frames = 0
        self.cooldown_threshold = 150  # 5 seconds at 30 FPS
        self.mouth_proximity_threshold = 0.25  # Distance threshold for hand near mouth (very lenient)
        self.head_tilt_threshold = 0.5  # Degrees - head tilted back when drinking (extremely sensitive)
        self.last_distance = 0  # For debugging
        self.last_head_tilt = 0  # For debugging

    def is_hand_near_mouth(self, hand_landmarks, face_landmarks):
        """Check if hand is near mouth region."""
        if not hand_landmarks or not face_landmarks:
            return False

        try:
            # Get mouth position (nose tip as proxy for mouth region)
            mouth_x = face_landmarks.landmark[1].x  # Nose tip
            mouth_y = face_landmarks.landmark[1].y

            # Get hand position (index finger tip)
            hand_x = hand_landmarks.landmark[8].x
            hand_y = hand_landmarks.landmark[8].y

            # Calculate normalized distance
            distance = m.sqrt((hand_x - mouth_x)**2 + (hand_y - mouth_y)**2)
            self.last_distance = distance  # Store for debugging

            return distance < self.mouth_proximity_threshold
        except Exception:
            return False

    def is_head_tilted_back(self, pose_landmarks):
        """Check if head is tilted back (common when drinking)."""
        if not pose_landmarks:
            return False

        try:
            lmPose = mp.solutions.pose.PoseLandmark

            # Get nose and chin positions
            nose = pose_landmarks.landmark[lmPose.NOSE]
            left_ear = pose_landmarks.landmark[lmPose.LEFT_EAR]
            right_ear = pose_landmarks.landmark[lmPose.RIGHT_EAR]
            left_eye = pose_landmarks.landmark[lmPose.LEFT_EYE]

            # Average ear position
            avg_ear_y = (left_ear.y + right_ear.y) / 2

            # If nose is significantly higher than ears, head is tilted back
            # In normalized coordinates, lower y value means higher on screen
            head_tilt = (avg_ear_y - nose.y) * 100  # Scale up for easier threshold
            self.last_head_tilt = head_tilt  # Store for debugging

            return head_tilt > self.head_tilt_threshold
        except Exception:
            return False

    def update(self, hand_near_mouth, head_tilted_back):
        """Update drinking detection state.

        Requires both hand near mouth AND head tilted back to detect drinking.
        """
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return False

        # Require both conditions for drinking detection
        drinking_condition = hand_near_mouth and head_tilted_back

        if drinking_condition:
            self.drinking_frames += 1
        else:
            self.drinking_frames = 0
            self.drinking_event_active = False

        if (self.drinking_frames >= self.drinking_threshold and
            not self.drinking_event_active):
            self.drinking_event_active = True
            self.cooldown_frames = self.cooldown_threshold
            return True

        return False

# ============================================================================
# SESSION TRACKER CLASS
# ============================================================================

class SessionTracker:
    """Track user presence, screen time, and sessions."""

    def __init__(self):
        self.sessions = []
        self.current_session = None
        self.is_present = False
        self.absence_frames = 0
        self.presence_frames = 0
        self.absence_threshold = 450  # 15 seconds at 30 FPS
        self.presence_threshold = 60   # 2 seconds at 30 FPS
        self.absence_start_time = None  # Track when absence started for real-time countdown
        self.absence_duration_sec = 15  # Real-time absence threshold in seconds
        self.last_break_reminder = None
        self.last_water_reminder = None
        self.break_interval = 1800  # 30 minutes
        self.water_interval = 1800  # 30 minutes

    def start_session(self):
        """Start a new screen time session."""
        now = datetime.now()
        self.current_session = {
            'session_id': len(self.sessions) + 1,
            'date': now.strftime('%Y-%m-%d'),
            'date_str': now.strftime('%b %d, %Y'),
            'start_time': time.time(),
            'start_time_str': now.strftime('%H:%M:%S'),
            'end_time': None,
            'end_time_str': None,
            'duration': 0,
            'good_posture_time': 0,
            'bad_posture_time': 0,
            'break_reminders': 0,
            'water_reminders': 0,
            'water_intake_count': 0,
        }
        self.last_break_reminder = time.time()
        self.last_water_reminder = time.time()

    def end_session(self):
        """End current session."""
        if self.current_session:
            self.current_session['end_time'] = time.time()
            self.current_session['end_time_str'] = datetime.now().strftime('%H:%M:%S')
            self.current_session['duration'] = (
                self.current_session['end_time'] - self.current_session['start_time']
            )
            self.sessions.append(self.current_session.copy())
            self.current_session = None

    def update_presence(self, person_detected):
        """Update presence status."""
        if person_detected:
            self.presence_frames += 1
            self.absence_frames = 0
            self.absence_start_time = None  # Reset absence timer

            if not self.is_present and self.presence_frames >= self.presence_threshold:
                self.is_present = True
                self.start_session()
        else:
            self.absence_frames += 1
            self.presence_frames = 0

            # Start absence timer on first absence frame
            if self.is_present and self.absence_start_time is None:
                self.absence_start_time = time.time()

            # Check if absence duration has exceeded threshold (using real time)
            if self.is_present and self.absence_start_time is not None:
                elapsed = time.time() - self.absence_start_time
                if elapsed >= self.absence_duration_sec:
                    self.is_present = False
                    self.absence_start_time = None
                    self.end_session()

    def update_posture_time(self, good_posture, frame_time):
        """Update posture time."""
        if self.current_session:
            if good_posture:
                self.current_session['good_posture_time'] += frame_time
            else:
                self.current_session['bad_posture_time'] += frame_time

    def record_water_intake(self):
        """Record water drinking event."""
        if self.current_session:
            self.current_session['water_intake_count'] += 1
            self.last_water_reminder = time.time()

    def check_break_reminder(self):
        """Check break reminder."""
        if self.current_session and self.last_break_reminder:
            elapsed = time.time() - self.last_break_reminder
            if elapsed >= self.break_interval:
                self.current_session['break_reminders'] += 1
                self.last_break_reminder = time.time()
                return True
        return False

    def check_water_reminder(self):
        """Check water reminder."""
        if self.current_session and self.last_water_reminder:
            elapsed = time.time() - self.last_water_reminder
            if elapsed >= self.water_interval:
                self.current_session['water_reminders'] += 1
                self.last_water_reminder = time.time()
                return True
        return False

    def get_session_time(self):
        """Get current session duration."""
        if self.current_session:
            return time.time() - self.current_session['start_time']
        return 0

    def get_time_until_water_reminder(self):
        """Get time until next water reminder."""
        if self.current_session and self.last_water_reminder:
            elapsed = time.time() - self.last_water_reminder
            remaining = max(0, self.water_interval - elapsed)
            return remaining
        return 0

    def get_absence_remaining_time(self):
        """Get remaining time before session ends due to absence."""
        if self.is_present and self.absence_start_time is not None:
            elapsed = time.time() - self.absence_start_time
            remaining = max(0, self.absence_duration_sec - elapsed)
            return remaining
        return 0

    def get_daily_stats(self):
        """Get statistics for today's sessions."""
        if not self.sessions and not self.current_session:
            return None

        all_sessions = self.sessions.copy()
        if self.current_session:
            temp_session = self.current_session.copy()
            temp_session['duration'] = time.time() - temp_session['start_time']
            temp_session['end_time_str'] = "Ongoing"
            all_sessions.append(temp_session)

        total_time = sum(s['duration'] for s in all_sessions)
        total_good = sum(s['good_posture_time'] for s in all_sessions)
        total_bad = sum(s['bad_posture_time'] for s in all_sessions)
        total_water = sum(s['water_intake_count'] for s in all_sessions)

        return {
            'sessions': all_sessions,
            'total_sessions': len(all_sessions),
            'total_time': total_time,
            'total_good_posture': total_good,
            'total_bad_posture': total_bad,
            'total_water_intake': total_water,
            'posture_percentage': (total_good / total_time * 100) if total_time > 0 else 0
        }

# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoProcessor:
    """Process video stream with pose and water detection."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

        self.session_tracker = SessionTracker()
        self.water_tracker = WaterIntakeTracker()

        self.fps = 30
        self.frame_time = 1.0 / self.fps
        self.good_frames = 0
        self.bad_frames = 0

        # Colors
        self.colors = {
            'red': (50, 50, 255),
            'green': (127, 255, 0),
            'light_green': (127, 233, 100),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255),
            'white': (255, 255, 255),
            'orange': (0, 165, 255),
            'cyan': (255, 255, 0),
            'blue': (255, 127, 0)
        }

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_alert = {"break": None, "water": None, "posture": None}

    def process_frame(self, frame):
        """Process single frame."""
        if frame is None:
            return None, None

        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose and hands
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Check presence
        person_detected = (pose_results.pose_landmarks is not None)
        self.session_tracker.update_presence(person_detected)

        # Status overlay
        status_y = 30

        # Presence status with absence countdown
        if self.session_tracker.is_present:
            presence_text = "[PRESENT]"
            presence_color = self.colors['green']
        else:
            presence_text = "[ABSENT]"
            presence_color = self.colors['red']
        cv2.putText(image, presence_text, (10, status_y), self.font, 0.8, presence_color, 2)

        # Show absence counter if user is present but detection is lost
        remaining_time = self.session_tracker.get_absence_remaining_time()
        if remaining_time > 0:
            warning_text = f"Session ends in: {int(remaining_time)}s"
            cv2.putText(image, warning_text, (w - 250, status_y), self.font, 0.6, self.colors['orange'], 2)

        # Debug: Show detection status
        detection_status = "Person Detected: YES" if person_detected else "Person Detected: NO"
        detection_color = self.colors['green'] if person_detected else self.colors['red']
        cv2.putText(image, detection_status, (10, h - 20), self.font, 0.5, detection_color, 1)

        alerts = []

        if self.session_tracker.is_present:
            # Session time
            session_time = self.session_tracker.get_session_time()
            time_text = f"Screen Time: {format_time(session_time)}"
            cv2.putText(image, time_text, (10, status_y + 35), self.font, 0.7, self.colors['white'], 2)

            # Water intake
            if self.session_tracker.current_session:
                water_count = self.session_tracker.current_session['water_intake_count']
                water_text = f"Water Intake: {water_count}x"
                cv2.putText(image, water_text, (10, status_y + 70), self.font, 0.7, self.colors['cyan'], 2)

                # Next water reminder
                time_until_water = self.session_tracker.get_time_until_water_reminder()
                water_reminder_text = f"Next: {format_time(time_until_water)}"
                cv2.putText(image, water_reminder_text, (10, status_y + 105), self.font, 0.6, self.colors['cyan'], 2)

            # Check reminders
            current_time = time.time()
            if self.session_tracker.check_break_reminder():
                if not self.last_alert["break"] or (current_time - self.last_alert["break"]) > 60:
                    alerts.append("BREAK REMINDER: Take a 5-minute break!")
                    self.last_alert["break"] = current_time

            if self.session_tracker.check_water_reminder():
                if not self.last_alert["water"] or (current_time - self.last_alert["water"]) > 60:
                    alerts.append("HYDRATION REMINDER: Drink water!")
                    self.last_alert["water"] = current_time

        # Detect hand near mouth and head tilt
        hand_near_mouth = False
        head_tilted_back = False

        # Check head tilt
        if pose_results.pose_landmarks:
            head_tilted_back = self.water_tracker.is_head_tilted_back(pose_results.pose_landmarks)

        # Check hand near mouth
        if hands_results.multi_hand_landmarks and pose_results.pose_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
                )

                if self.water_tracker.is_hand_near_mouth(hand_landmarks, pose_results.pose_landmarks):
                    hand_near_mouth = True
                    break


        # Check for drinking event
        drinking_detected = self.water_tracker.update(hand_near_mouth, head_tilted_back)
        if drinking_detected and self.session_tracker.is_present:
            self.session_tracker.record_water_intake()
            cv2.rectangle(image, (0, 0), (w, h), self.colors['cyan'], 15)
            alerts.append("Water intake recorded!")

        # Process posture
        if person_detected and self.session_tracker.is_present:
            lm = pose_results.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            try:
                # Get coordinates
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
                l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
                l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

                # Alignment
                offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
                align_text = f"Align: {int(offset)}"
                align_color = self.colors['green'] if offset < 120 else self.colors['red']
                cv2.putText(image, align_text, (w - 150, 70), self.font, 0.6, align_color, 2)

                # Calculate angles
                neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
                torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

                # Posture determination
                good_posture = (neck_inclination < 35 and torso_inclination < 15)  # Adjusted neck threshold to 35 degrees

                # Update tracking
                self.session_tracker.update_posture_time(good_posture, self.frame_time)

                if good_posture:
                    self.bad_frames = 0
                    self.good_frames += 1
                    posture_color = self.colors['light_green']
                    posture_text = "GOOD POSTURE"
                else:
                    self.good_frames = 0
                    self.bad_frames += 1
                    posture_color = self.colors['red']
                    posture_text = "BAD POSTURE"

                # Display posture info (moved down to avoid overlap with water indicators)
                posture_y_offset = 180 if (hand_near_mouth or head_tilted_back) else 140
                cv2.putText(image, posture_text, (10, status_y + posture_y_offset),
                           self.font, 0.8, posture_color, 2)
                angle_text = f"Neck: {int(neck_inclination)}deg | Torso: {int(torso_inclination)}deg"
                cv2.putText(image, angle_text, (10, status_y + posture_y_offset + 35),
                           self.font, 0.6, posture_color, 2)

                # Draw skeleton
                line_color = self.colors['green'] if good_posture else self.colors['red']
                cv2.circle(image, (l_shldr_x, l_shldr_y), 8, self.colors['yellow'], -1)
                cv2.circle(image, (l_ear_x, l_ear_y), 8, self.colors['yellow'], -1)
                cv2.circle(image, (l_hip_x, l_hip_y), 8, self.colors['yellow'], -1)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), line_color, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), line_color, 4)

                # Bad posture warning
                bad_time = self.bad_frames * self.frame_time
                if bad_time > 300:  # 5 minutes
                    current_time = time.time()
                    if not self.last_alert["posture"] or (current_time - self.last_alert["posture"]) > 10:
                        alerts.append("WARNING: Bad posture for 5+ minutes!")
                        self.last_alert["posture"] = current_time
                    self.bad_frames = 0

            except Exception as e:
                pass

        return image, alerts

    def get_stats(self):
        """Get current statistics."""
        return self.session_tracker.get_daily_stats()

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

class HealthMonitorApp:
    """Gradio application for health monitoring."""

    def __init__(self, camera_index=0):
        self.processor = VideoProcessor()
        self.cap = None
        self.running = False
        self.alert_queue = queue.Queue()
        self.camera_index = int(camera_index)

    def start_camera(self):
        """Start camera capture using configured camera index."""
        if self.cap is None:
            # Helper to try opening a camera index robustly
            def _try_open(idx):
                try:
                    cap_test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                except Exception:
                    cap_test = cv2.VideoCapture(idx)
                if cap_test is not None and cap_test.isOpened():
                    return cap_test
                try:
                    if cap_test is not None:
                        cap_test.release()
                except Exception:
                    pass
                return None

            # First try configured index
            cap = _try_open(self.camera_index)

            # If configured index fails, probe common indices preferring non-zero (external webcams)
            if cap is None:
                probe_indices = list(range(1, 8)) + [0]
                for idx in probe_indices:
                    if idx == self.camera_index:
                        continue
                    cap = _try_open(idx)
                    if cap is not None:
                        self.camera_index = idx
                        break

            if cap is None:
                return f"[ERROR] Unable to open camera index {self.camera_index} or probe indices"

            # assign and mark running
            self.cap = cap
            self.running = True
            return f"[OK] Camera started (index={self.camera_index})"
        return "[WARNING] Camera already running"

    def stop_camera(self):
        """Stop camera capture."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return "[STOPPED] Camera stopped"

    def video_stream(self):
        """Generate video frames."""
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            processed_frame, alerts = self.processor.process_frame(frame)

            if alerts:
                for alert in alerts:
                    self.alert_queue.put(alert)

            if processed_frame is not None:
                yield cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            time.sleep(0.03)

    def get_session_table(self):
        """Generate session table HTML."""
        stats = self.processor.get_stats()

        if not stats:
            return "<p style='text-align:center; color:#888;'>No sessions recorded yet</p>"

        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        html += "<table style='width:100%; border-collapse: collapse;'>"
        html += "<thead style='background:#2d2d2d; position:sticky; top:0;'>"
        html += "<tr>"
        html += "<th style='padding:10px; border:1px solid #444;'>Session</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>Date</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>Start</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>End</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>Duration</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>Good Posture</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>Bad Posture</th>"
        html += "<th style='padding:10px; border:1px solid #444;'>💧 Water</th>"
        html += "</tr></thead><tbody>"

        for session in stats['sessions']:
            duration_pct = (session['duration'] / stats['total_time'] * 100) if stats['total_time'] > 0 else 0
            good_pct = (session['good_posture_time'] / session['duration'] * 100) if session['duration'] > 0 else 0

            status = "🟢 Ongoing" if session.get('end_time_str') == "Ongoing" else "⚪"
            date_display = session.get('date_str', session.get('date', 'N/A'))

            html += f"<tr style='background:{'#1a1a1a' if session['session_id'] % 2 == 0 else '#252525'};'>"
            html += f"<td style='padding:10px; border:1px solid #444;'>{status} #{session['session_id']}</td>"
            html += f"<td style='padding:10px; border:1px solid #444; color:#FFD700;'>{date_display}</td>"
            html += f"<td style='padding:10px; border:1px solid #444;'>{session['start_time_str']}</td>"
            html += f"<td style='padding:10px; border:1px solid #444;'>{session.get('end_time_str', 'Ongoing')}</td>"
            html += f"<td style='padding:10px; border:1px solid #444;'>{format_time(session['duration'])}</td>"
            html += f"<td style='padding:10px; border:1px solid #444; color:#7FFF00;'>{format_time(session['good_posture_time'])} ({good_pct:.0f}%)</td>"
            html += f"<td style='padding:10px; border:1px solid #444; color:#FF6B6B;'>{format_time(session['bad_posture_time'])}</td>"
            html += f"<td style='padding:10px; border:1px solid #444; color:#00D9FF;'>{session['water_intake_count']}x</td>"
            html += "</tr>"

        html += "</tbody></table></div>"
        return html

    def get_daily_summary(self):
        """Generate daily summary."""
        stats = self.processor.get_stats()

        if not stats:
            return "No data available yet. Start a session to see statistics."

        summary = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
            <h2 style='margin-top:0;'>📊 Today's Summary</h2>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;'>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 14px; opacity: 0.9;'>Total Sessions</div>
                    <div style='font-size: 32px; font-weight: bold;'>{stats['total_sessions']}</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 14px; opacity: 0.9;'>Total Screen Time</div>
                    <div style='font-size: 32px; font-weight: bold;'>{format_time(stats['total_time'])}</div>
                </div>
                <div style='background: rgba(127,255,0,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 14px; opacity: 0.9;'>Good Posture</div>
                    <div style='font-size: 32px; font-weight: bold;'>{format_time(stats['total_good_posture'])}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>{stats['posture_percentage']:.1f}% of total time</div>
                </div>
                <div style='background: rgba(255,107,107,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 14px; opacity: 0.9;'>Bad Posture</div>
                    <div style='font-size: 32px; font-weight: bold;'>{format_time(stats['total_bad_posture'])}</div>
                </div>
                <div style='background: rgba(0,217,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 14px; opacity: 0.9;'>💧 Water Intake</div>
                    <div style='font-size: 32px; font-weight: bold;'>{stats['total_water_intake']}x</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 14px; opacity: 0.9;'>Avg per Session</div>
                    <div style='font-size: 32px; font-weight: bold;'>{stats['total_water_intake'] / stats['total_sessions']:.1f}x</div>
                </div>
            </div>
        </div>
        """
        return summary

    def get_alerts(self):
        """Get recent alerts."""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
            except:
                break

        if not alerts:
            return ""

        alert_html = "<div style='background:#ff6b6b; padding:15px; border-radius:8px; margin:10px 0;'>"
        for alert in alerts[-3:]:  # Show last 3 alerts
            alert_html += f"<div style='margin:5px 0;'><b>{alert}</b></div>"
        alert_html += "</div>"
        return alert_html

    def get_all_stats(self):
        """Get all stats at once for updating."""
        return (
            self.get_daily_summary(),
            self.get_session_table(),
            self.get_alerts()
        )

    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(theme=gr.themes.Soft(), title="Health Monitor") as demo:
            gr.Markdown("""
            # 🏥 Posture, Hydration & Screen Time Monitor
            Monitor your health metrics in real-time with AI-powered tracking
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    video_output = gr.Image(label="Live Monitor", streaming=True)

                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                        stop_btn = gr.Button("⏸️ Stop Monitoring", variant="stop")

                    camera_status = gr.Textbox(label="Status", value="Ready to start", interactive=False)

                with gr.Column(scale=1):
                    alerts_display = gr.HTML(label="🔔 Recent Alerts")
                    summary_display = gr.HTML(label="📊 Daily Summary")

            gr.Markdown("## 📋 Session History")
            session_table = gr.HTML(label="Sessions")

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")

            # Event handlers
            def start_stream():
                status = self.start_camera()
                return status

            def stop_stream():
                status = self.stop_camera()
                return status

            def refresh_stats():
                """Manually refresh all statistics."""
                return self.get_all_stats()

            start_btn.click(
                fn=start_stream,
                outputs=camera_status
            ).then(
                fn=self.video_stream,
                outputs=video_output
            )

            stop_btn.click(
                fn=stop_stream,
                outputs=camera_status
            )

            refresh_btn.click(
                fn=refresh_stats,
                outputs=[summary_display, session_table, alerts_display]
            )

            # Initial load of stats
            demo.load(
                fn=refresh_stats,
                outputs=[summary_display, session_table, alerts_display]
            )

            # Auto-refresh stats every 3 seconds using Gradio's built-in timer
            timer = gr.Timer(value=3, active=True)
            timer.tick(
                fn=refresh_stats,
                outputs=[summary_display, session_table, alerts_display]
            )

        return demo

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Health Monitor - Hand + Head Tilt Detection')
    default_cam = int(os.environ.get('CAMERA_INDEX', 0))
    parser.add_argument('--camera', type=int, default=default_cam, help='Camera index to use (default from CAMERA_INDEX env or 0)')
    args = parser.parse_args()

    app = HealthMonitorApp(camera_index=args.camera)
    demo = app.create_interface()
    demo.launch(share=True, server_port=7860)

if __name__ == "__main__":
    print("="*60)
    print("🏥 Health Monitor - Starting Gradio Interface")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Real-time posture detection")
    print("  ✓ Water intake tracking (hand + head tilt detection)")
    print("  ✓ Screen time monitoring")
    print("  ✓ Session-based statistics")
    print("  ✓ Break & hydration reminders")
    print("\nInstructions:")
    print("  1. Click 'Start Monitoring' to begin")
    print("  2. Position yourself for side view (best for posture)")
    print("  3. Bring hand near mouth AND tilt head back to log water intake")
    print("  4. View real-time stats in the dashboard")
    print("\nWater Detection Method:")
    print("  → Hand near mouth detection")
    print("  → Head tilt back detection")
    print("  → Both conditions required for water intake logging")
    print("="*60)

    main()
