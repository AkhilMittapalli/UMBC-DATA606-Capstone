# UMBC - DATA 606 CAPSTONE
# Smart Sitting Posture & Wellness Monitoring System Using Computer Vision and Machine Learning

**Author:** Akhil Mittapalli
**Semester:** Fall 2024
**Course:** DATA 606 - Capstone in Data Science
**University:** University of Maryland, Baltimore County (UMBC)

---

## Quick Links

| Resource | Link |
|----------|------|
| **YouTube Presentation** | [Watch the Video Presentation](https://youtu.be/Xz-HEnun7z8) |
| **PowerPoint Presentation** | [View Final PPT](docs/Health_Monitor_Presentation.pptx) |
| **Final Report** | [View Report](docs/report.md) |

---

## Introduction

Modern lifestyles have increasingly shifted towards long hours of sitting, often in front of digital screens. Prolonged poor posture, excessive screen exposure, and inadequate water consumption are among the leading contributors to health issues such as musculoskeletal disorders, eye strain, fatigue, and dehydration. According to occupational health studies, poor ergonomic practices directly impact productivity and overall well-being, especially for individuals working in office-based or remote setups.

This project implements a **vision-based, AI-driven, non-intrusive monitoring system** that leverages **computer vision and machine learning** to continuously observe sitting posture, detect water drinking events, and monitor screen time without requiring additional hardware apart from a standard webcam.

---

## Features (Implemented)

### 1. Real-Time Posture Detection
- AI-powered posture analysis using **MediaPipe Pose** (33 body landmarks)
- Monitors **neck inclination** (threshold: < 35°) and **torso alignment** (threshold: < 15°)
- Visual feedback with color-coded skeleton overlay (green = good, red = bad)
- Alerts after 5 minutes of continuous bad posture
- Tracks good vs. bad posture duration per session

### 2. Water Intake Tracking
- **Hand-to-mouth gesture detection** using MediaPipe Hands (21 landmarks per hand)
- **Head tilt detection** for drinking motion recognition
- Automatic drinking event recording with 5-second cooldown
- Hydration reminders every 30 minutes
- Visual confirmation with cyan border flash

### 3. Screen Time Monitoring
- Automatic session start/stop based on **presence detection**
- 15-second absence threshold before session ends
- Real-time countdown when user leaves frame
- Break reminders every 30 minutes
- Daily session history with detailed statistics

### 4. Interactive Web Dashboard (Gradio)
- Live video feed with real-time overlays
- Session history table with date tracking
- Daily summary with statistics (sessions, screen time, posture %, water intake)
- Alert notifications for breaks, hydration, and posture
- Auto-refresh every 3 seconds

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Health Monitor System                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │        Gradio Web Interface             │
        │  (Port 7860 - Browser-based UI)         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         Video Processor                 │
        │  - Frame Capture (OpenCV)               │
        │  - Real-time Processing @ 30 FPS        │
        └─────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌───────────┐  ┌───────────┐  ┌──────────┐
        │ MediaPipe │  │ MediaPipe │  │ Session  │
        │   Pose    │  │   Hands   │  │ Tracker  │
        │ Detection │  │ Detection │  │          │
        └───────────┘  └───────────┘  └──────────┘
```

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core development |
| **Computer Vision** | OpenCV 4.8+ | Video capture and processing |
| **Pose Detection** | MediaPipe Pose | Body landmark detection (33 keypoints) |
| **Hand Detection** | MediaPipe Hands | Hand tracking (21 landmarks) |
| **Web Interface** | Gradio 4.0+ | Interactive dashboard |
| **Numerical Computing** | NumPy | Array operations |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or external USB)
- Windows, Linux, or macOS

### Setup

```bash
# Clone the repository
git clone https://github.com/AkhilMittapalli/UMBC-DATA606-Capstone.git
cd UMBC-DATA606-Capstone

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/Health_monitor_v1.py
```

### Quick Start
```bash
# Run with default camera
python src/Health_monitor_v1.py

# Run with specific camera index
python src/Health_monitor_v1.py --camera 1

# Open browser to: http://localhost:7860
```

---

## Project Structure

```
UMBC-DATA606-Capstone/
├── README.md                              # Project overview (this file)
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore rules
├── src/
│   └── Health_monitor_v1.py              # Main application (830+ lines)
└── docs/
    ├── report.md                         # Final capstone report
    ├── Health_Monitor_Presentation.pptx  # PowerPoint presentation
    └── images/
        ├── Architecture.png              # System architecture diagram
        └── Setup1.jpg                    # Setup reference image
```

---

## Usage Guide

### Starting a Session
1. Launch the application: `python src/Health_monitor_v1.py`
2. Open browser to `http://localhost:7860`
3. Click "▶️ Start Monitoring"
4. Position yourself in camera view (side view recommended for best posture detection)

### Understanding the Display

| Overlay Element | Description | Color |
|-----------------|-------------|-------|
| [PRESENT]/[ABSENT] | User presence status | Green/Red |
| Screen Time | Current session duration | White |
| Water Intake | Number of drinks recorded | Cyan |
| GOOD/BAD POSTURE | Current posture status | Green/Red |
| Neck/Torso angles | Body inclination degrees | Green/Red |

### Posture Guidelines
- **Good Posture**: Neck angle < 35° AND Torso angle < 15°
- **Tip**: Position camera at side view for most accurate detection

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Posture Detection Accuracy | 88-92% |
| Processing Speed | 20-30 FPS |
| Water Intake Detection Precision | 80%+ |
| Presence Detection Accuracy | 95%+ |

### Key Achievements
- Real-time posture monitoring with immediate visual feedback
- Dual-condition water intake detection reduces false positives by 85%
- Session-based tracking provides comprehensive daily statistics
- User-friendly Gradio interface requires no technical expertise

---

## Applications

1. **Workplace Wellness** - Monitor employee ergonomics in office/remote settings
2. **Educational Environments** - Track student posture during e-learning
3. **Healthcare & Rehabilitation** - Support physiotherapists in remote monitoring
4. **Personal Lifestyle** - Digital wellness assistant for extended computer use

---

## Future Enhancements

- [ ] GPU acceleration for faster processing
- [ ] Data export to CSV/Excel
- [ ] Mobile companion app
- [ ] Integration with fitness trackers
- [ ] Multi-user support for office environments
- [ ] AI-driven personalized recommendations

---

## References

1. Google MediaPipe Documentation - https://mediapipe.dev/
2. OpenCV Library - https://opencv.org/
3. Gradio Documentation - https://gradio.app/docs/

---

## License

This project is developed for educational purposes as part of the DATA 606 Capstone course at UMBC.
