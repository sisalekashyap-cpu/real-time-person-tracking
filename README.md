# Real-Time Person Detection & Tracking

A real-time computer vision project for detecting and tracking people in video using **OpenCV, MobileNet-SSD, and centroid-based object tracking**.

> **Important:** This project performs person detection and temporary object tracking. It does **not** perform facial recognition or identify people by their real-world identity.

---

## Overview

This project processes video frames in real time, detects people using the **MobileNet-SSD** object detection model, and tracks detected people across frames using a **centroid-based tracking algorithm**.

The system can:

* Detect people in video
* Draw bounding boxes around detected people
* Assign temporary tracking IDs
* Maintain tracking IDs across frames
* Count currently tracked people
* Display tracking information and FPS
* Process video files and camera input
* Configure detection and tracking parameters

This repository began as a project from an AI/GenAI workshop and was later modified and extended as a personal computer vision project.

---

## How It Works

The basic processing pipeline is:

```text
Video / Camera
      │
      ▼
 Read Frame
      │
      ▼
Pre-processing
      │
      ▼
 MobileNet-SSD
      │
      ▼
Person Detection
      │
      ▼
Bounding Boxes
      │
      ▼
Non-Maximum Suppression
      │
      ▼
Centroid Tracker
      │
      ▼
Temporary Tracking IDs
      │
      ▼
Person Count / Visualization
```

### 1. Person Detection

Each video frame is passed through the MobileNet-SSD model using OpenCV's DNN module.

The detector produces:

* Object class
* Confidence score
* Bounding-box coordinates

Only detections classified as `person` are processed by the tracking system.

### 2. Object Tracking

The centroid tracker calculates the center point of each detected bounding box and compares it with previously tracked objects.

This allows the system to maintain a temporary ID as a person moves between frames.

For example:

```text
Frame 1 → Person → ID 0
Frame 2 → Person → ID 0
Frame 3 → Person → ID 0
Frame 4 → Person → ID 0
```

When a new object appears, the tracker assigns a new ID.

---

## Tracking IDs Are Not Real Identities

The tracking IDs displayed by this application are temporary identifiers assigned by the tracking algorithm.

For example:

```text
ID: 0
ID: 1
ID: 2
```

These do **not** represent:

```text
Student 0
Student 1
Student 2
```

The system does not recognize who a person is.

| Capability             | Supported |
| ---------------------- | --------: |
| Person detection       |         ✅ |
| Object tracking        |         ✅ |
| Temporary tracking IDs |         ✅ |
| Person counting        |         ✅ |
| Facial recognition     |         ❌ |
| Identity recognition   |         ❌ |
| Name-based attendance  |         ❌ |

This distinction is important because object tracking and identity recognition are different computer-vision problems.

---

## Features

### Detection

* MobileNet-SSD person detection
* Configurable confidence threshold
* Bounding-box visualization
* Non-Maximum Suppression

### Tracking

* Centroid-based object tracking
* Temporary tracking IDs
* Configurable disappearance threshold
* Configurable tracking distance

### Input

The application supports configurable video input, including:

* Video files
* Webcam / camera input
* Supported video streams

### Runtime Information

The application can display:

* Current FPS
* Active tracked people
* Total unique tracking IDs observed

---

## Technology Stack

| Component                | Technology        |
| ------------------------ | ----------------- |
| Programming Language     | Python            |
| Computer Vision          | OpenCV            |
| Object Detection         | MobileNet-SSD     |
| Tracking                 | Centroid Tracking |
| Neural Network Interface | OpenCV DNN        |
| Numerical Processing     | NumPy             |
| Video Processing         | OpenCV            |

---

## Project Structure

```text
real-time-person-tracking/
│
├── person_tracking.py
├── centroidtracker.py
│
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
│
├── test_video.mp4
├── requirements.txt
├── README.md
└── .gitignore
```

### File Descriptions

| File                             | Description                                    |
| -------------------------------- | ---------------------------------------------- |
| `person_tracking.py`             | Main person detection and tracking application |
| `centroidtracker.py`             | Centroid-based object tracking implementation  |
| `MobileNetSSD_deploy.prototxt`   | MobileNet-SSD network architecture             |
| `MobileNetSSD_deploy.caffemodel` | Trained MobileNet-SSD model                    |
| `test_video.mp4`                 | Example test video                             |
| `requirements.txt`               | Python dependencies                            |
| `.gitignore`                     | Files excluded from version control            |

---

## Requirements

* Python 3.10+
* OpenCV
* NumPy
* SciPy
* imutils
* Webcam or video input

The project is designed to run on CPU with a standard OpenCV installation.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sisalekashyap-cpu/real-time-person-tracking.git
cd real-time-person-tracking
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the application with:

```bash
python person_tracking.py
```

The application can be configured through its available command-line options for input source, detection confidence, tracking parameters, and other runtime settings.

---

## Understanding the Tracker

The centroid tracker uses the center of each bounding box:

```text
(x1, y1)
   ┌───────────────┐
   │               │
   │       ●       │
   │    centroid   │
   │               │
   └───────────────┘
             (x2, y2)
```

The centroid is calculated as:

```text
cX = (x1 + x2) / 2
cY = (y1 + y2) / 2
```

The tracker then compares current centroids with previously tracked centroids to determine which detections correspond to existing objects.

---

## Limitations

This project uses a lightweight detection and tracking approach, so tracking is not perfect in every situation.

Tracking IDs can change when:

* A person moves very quickly
* People overlap heavily
* A person becomes partially or completely occluded
* Detection confidence drops
* Lighting changes significantly
* Bounding boxes change substantially between frames
* Multiple people cross paths

The system should therefore be considered an **object tracking system**, not an identity tracking or facial recognition system.

### Detection limitations

MobileNet-SSD is a relatively lightweight object detector. Detection performance can vary depending on:

* Lighting conditions
* Camera quality
* Camera angle
* Distance from the camera
* Person size within the frame
* Occlusion
* Video resolution

---

## What I Changed

The original workshop implementation provided the foundation for this project.

The original implementation demonstrated:

* OpenCV DNN-based detection
* MobileNet-SSD
* Person filtering
* Bounding boxes
* Centroid tracking
* Temporary tracking IDs
* Person counting

I subsequently modified and extended the project with improvements including:

* Configurable input sources
* Command-line configuration
* Detection confidence configuration
* Tracking parameter configuration
* Non-Maximum Suppression using OpenCV
* Improved input handling
* Runtime FPS information
* Tracking visualization
* Additional runtime controls
* Updated project documentation

The goal of this repository is to document my work with real-time computer vision while clearly acknowledging the original project that served as the starting point.

---

## Original Workshop Project

This project originated from an AI/GenAI workshop implementation based on the following repository:

**AI-Powered Attendance System for Corporate Universities**

Original repository:

https://github.com/theyashkhatri/AI-Powered-Attendence-System-for-Corporate-Universities

The original project was the starting point for the detection and tracking implementation used here.

This repository is a modified version of that starting point and is maintained separately to document my own changes and experimentation.

---

## Why This Repository Uses "Person Tracking"

The original project uses the term **"Attendance System"**, but the underlying computer-vision pipeline detects and tracks people as objects.

It does not:

* Recognize a person's face
* Determine a person's name
* Match a person to a student or employee database
* Verify someone's identity
* Record attendance against a real identity

For that reason, this repository is described as:

> **Real-Time Person Detection & Tracking**

rather than claiming to provide real identity-based attendance.

---

## Future Improvements

Possible future work includes:

* Modern object detection models
* More robust multi-object tracking
* Improved occlusion handling
* Trajectory visualization
* Entry/exit counting
* Region-of-interest analytics
* Dwell-time analysis
* Heatmaps
* CSV analytics
* Web dashboard
* GPU acceleration
* Performance benchmarking

---

## Learning Outcomes

This project provided practical experience with:

* Computer vision
* Object detection
* OpenCV DNN
* MobileNet-SSD
* Bounding boxes
* Confidence thresholds
* Non-Maximum Suppression
* Object tracking
* Centroid tracking
* Video processing
* Real-time FPS measurement
* Python project organization

---

## License

No separate license has currently been added to this repository.

Because this project originated from an existing workshop implementation, please refer to the **original repository** and its applicable terms before redistributing or reusing material derived from it.

---

## Author

**Kashyap Sisale**

GitHub:
https://github.com/sisalekashyap-cpu
