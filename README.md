# Smart-Frames-Optimized-Detection-for-People-and-Faces

## Overview
Smart Frames is an AI-powered, optimized detection system designed for real-time analysis of high-quality images and videos. By leveraging frame skipping, multithreading, and sequential decision-making algorithms, this system efficiently detects people, performs face detection, and runs face recognition only when needed—minimizing computation on resource-constrained hardware.

This project can be adapted for multiple video streams and optimized for various hardware environments.

## Features
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
- **Retail Analytics**: Track customers and perform demographic analysis using people detection.

## Getting Started

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.8 or above
- OpenCV
- YOLOv8 (for people detection)
- DLIB (for face recognition)
- Multithreading Libraries

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Smart-Frames.git
   cd Smart-Frames
