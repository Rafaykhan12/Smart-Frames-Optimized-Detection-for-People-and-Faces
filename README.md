# Smart-Frames-Optimized-Detection-for-People-and-Faces

## Overview
Smart Frames is an AI-powered solution designed to detect and count people in a designated area and recognize whether they are known individuals. The system first identifies if there are people present, then checks for face detection, and finally performs face recognition if faces are found. Built with efficiency in mind, Smart Frames optimizes real-time analysis of high-quality images and videos by incorporating frame skipping and multithreading, ensuring the system only performs computationally intensive tasks when necessary. This approach reduces processing overhead, making it suitable for resource-constrained hardware while maintaining high accuracy and responsiveness.

This project can be adapted for multiple video streams and optimized for various hardware environments.

## Features
- **People Detection**: Efficiently detects if people are present
- **Face Detection**: Once people are detected, the system checks for faces within the frame.
- **Face Recognition**: If faces are found, it then performs recognition to identify known individuals.
- **Frame Skipping**: Skip unnecessary frames to save computation without sacrificing accuracy.
- **Sequential Detection**:
  - Detect people in frames first.
  - Only perform face detection if people are found.
  - Run face recognition after face detection.
- **Multithreading**: Parallel processing of frames for real-time detection on multiple streams.
- **Optimized for High-Quality Images**: Designed to handle multiple streams of high-resolution video.

## How It Works
1. **Frame Capture**: The system captures and processes frames from video streams.
2. **People Detection**: If no person is found, the system skips face detection and recognition, saving resources.
3. **Face Detection**: If people are detected, it checks for faces within bounding boxes.
4. **Face Recognition**: If faces are detected, the system recognizes them using pre-trained models.
5. **Results Storage**: Frames with detected people are saved for further face processing.

## Use Cases
- **Smart Surveillance**: Deployable in smart security systems to monitor people and perform face recognition in real-time.
- **Crowd Analysis**: Detect people in crowds and identify specific individuals for security or event monitoring.
- **Industrial, Warehouse, and Hospital Monitoring**: Applicable in industries, warehouses, hospitals, and other locations where it's necessary to detect the presence of people and identify whether they are known or unknown for security and access control.

## Getting Started

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.8 or above
- OpenCV
- YOLOv5 (for people detection)
- face_recognition (for face recognition)
- Multithreading Libraries

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rafaykhan12/Smart-Frames-Optimized-Detection-for-People-and-Faces
   cd Smart-Frames

2. Install Dependencies
Ensure that you have Python 3.8 or higher installed. To install the required packages, run:

```bash
   pip install -r requirements.txt
```


### Usage

### Run the System
To start the main detection pipeline, use the following command:

```bash
python main.py
