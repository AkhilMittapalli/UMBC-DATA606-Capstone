# UMBC - DATA 606 CAPSTONE
# Smart Sitting Posture & Wellness Monitoring System Using Computer Vision and Machine Learning

## Introduction
Modern lifestyles have increasingly shifted towards long hours of sitting, often in front of digital screens. Prolonged poor posture, excessive screen exposure, and inadequate water consumption are among the leading contributors to health issues such as musculoskeletal disorders, eye strain, fatigue, and dehydration. According to occupational health studies, poor ergonomic practices directly impact productivity and overall well-being, especially for individuals working in office-based or remote setups.

Traditional solutions such as posture reminder devices, manual hydration apps, or external screen-time trackers address these issues in isolation and often require active user input. They are either intrusive, expensive, or not integrated into daily workflows.

This project proposes a **vision-based, AI-driven, non-intrusive monitoring system** that leverages **computer vision and machine learning** to continuously observe sitting posture, detect water drinking events, and monitor screen time without requiring additional hardware apart from a standard webcam. By integrating these three wellness dimensions into a unified system, the project aims to create a proactive assistant that not only alerts users in real time but also provides actionable insights to encourage long-term healthy habits.

---

## Objectives
1. **Posture Monitoring:** Develop a machine learning model to detect and classify sitting postures as good or bad in real time.  
2. **Hydration Tracking:** Detect water drinking activity automatically using pose estimation and object detection, eliminating the need for manual water intake logging.  
3. **Screen Time Monitoring:** Track screen exposure duration by detecting user presence through face recognition techniques.  
4. **Holistic Wellness Alerts:** Provide timely alerts and visual feedback to help users correct posture, take breaks, and stay hydrated.  

---

## Methodology

### 1. Posture Recognition (Custom ML Model)
- **Dataset:** The project will utilize a labeled dataset from Roboflow (*Sitting Posture Dataset*), covering multiple posture categories such as correct sitting, slouching, and leaning forward. Additional images may be collected to balance classes and improve robustness.  
- **Preprocessing:** Images will be resized, normalized, and augmented (rotation, scaling, brightness adjustments) to improve model generalization.  
- **Model Training:**  
  - A YOLOv8-based model will be fine-tuned for real-time posture detection and classification.  
  - The model will output bounding boxes and posture labels directly on live webcam feeds.  
- **Integration:** Posture classification results will be continuously logged. If poor posture persists beyond a set threshold (e.g., 30 seconds), the system will issue alerts via visual or audio notifications.  

### 2. Water Drinking Detection (Hybrid CV + Object Detection Approach)
- **MediaPipe Pose + Hands:**  
  - Tracks key body and hand landmarks.  
  - Detects hand motion from the lower body region toward the mouth region.  
  - If hand remains near the mouth for a sustained period, the system flags a potential drinking event.  
- **YOLOv8 Object Detection (Bottle/Cup):**  
  - A pretrained YOLOv8 detection model will be integrated to detect bottles or cups in the user’s hand.  
  - Drinking is confirmed when the bottle overlaps with the mouth region, reducing false positives (e.g., eating, scratching face).  
- **Event Logging:** Confirmed drinking events are automatically logged with timestamps, and cumulative daily water intake is estimated.  

### 3. Screen Time Monitoring (Face Presence Detection)
- **MediaPipe Face Detection:**  
  - Detects the user’s face in front of the webcam.  
  - Presence initiates a timer, while absence pauses it.  
- **Extended Monitoring:**  
  - The system ensures the correct user is detected by integrating optional face recognition modules to differentiate between individuals.  
  - Eye state detection may also be added to ensure that the user is actively engaged with the screen rather than just present.  
- **Reminders:** If continuous screen usage exceeds recommended limits (e.g., one hour), the system generates break reminders.  

### 4. Alerts, Feedback, and Dashboard
- **Alerts:**  
  - Real-time posture correction reminders.  
  - Break alerts for prolonged screen exposure.  
  - Hydration reminders if no drinking event is logged within a set timeframe.  
- **Dashboard (Streamlit-based):**  
  - Displays posture statistics (duration in good vs bad posture).  
  - Shows cumulative screen time across the day.  
  - Tracks hydration frequency with drinking event logs.  
  - Provides weekly and monthly summaries for long-term wellness analysis.  

---

## Technology Stack

### **Programming Language**
- **Python 3.10+**

### **Machine Learning & Computer Vision**
- **YOLOv8 (Ultralytics)** → Posture classification, bottle/cup detection  
- **MediaPipe (Google)** → Pose, Hands, Face detection  
- **OpenCV** → Video capture and visualization  

### **Data Handling & Training**
- **PyTorch** → Model training and fine-tuning  
- **Pandas / NumPy** → Data logging and analysis  

### **Visualization & Dashboard**
- **Matplotlib / Seaborn** → Analytics and visualization  
- **Streamlit** → Interactive wellness dashboard  

### **Notifications**
- **Pygame / Toast / Custom Popups** → Real-time user alerts  

---

## Expected Outcomes
- A robust computer vision system capable of detecting **good vs bad posture** with high accuracy.  
- An automated water drinking detection mechanism with minimized false positives.  
- Accurate real-time monitoring of screen exposure duration.  
- A unified wellness assistant that combines posture, hydration, and screen-time awareness into a single platform.  

---

## Applications
1. **Workplace Wellness**  
   - Monitor and improve employee ergonomics in corporate or remote work settings.  
   - Prevent musculoskeletal strain and improve productivity.  

2. **Educational Environments**  
   - Track posture and screen habits of students in classrooms or e-learning setups.  
   - Encourage healthier digital learning practices.  

3. **Healthcare & Rehabilitation**  
   - Support physiotherapists in monitoring patient posture remotely.  
   - Aid in recovery programs for individuals with chronic back or neck issues.  

4. **Personal Lifestyle Improvement**  
   - Serve as a digital wellness assistant for individuals who spend extended hours on computers.  
   - Provide personalized health insights over time.  
---

## Future Scope
- **IoT Integration:** Combine with smart chairs (for weight/pressure distribution) and smart water bottles to improve accuracy.  
- **Multi-user Support:** Extend the system to track multiple users simultaneously in office or classroom settings.  
- **Mobile/Edge Deployment:** Deploy lightweight versions of the system on smartphones or edge devices (e.g., Raspberry Pi) for portability.  
- **Advanced Action Recognition:** Replace heuristic drinking detection with a fine-tuned action recognition model trained on large-scale video datasets.  
- **AI-driven Personalization:** Implement adaptive ML models that learn user-specific habits and generate personalized wellness recommendations.  
- **Healthcare Integration:** Sync with electronic health record (EHR) systems for remote patient monitoring in occupational therapy and ergonomics clinics.  
