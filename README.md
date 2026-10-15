# YOLOv8 Waste Detection

A waste detection system using YOLOv8 to detect and classify organic and inorganic waste in real time through a webcam, integrated with a Flask web application.

## 📌 Project Overview

This project aims to detect organic and inorganic waste using computer vision and deep learning. The YOLOv8 model is trained on a custom-labeled dataset and integrated into a Flask web application to enable real-time detection through a webcam.

The application provides a simple web interface with login functionality, a detection page, and controls to start or stop detection.

## 🎯 Objectives

- Detect waste objects using YOLOv8.
- Classify detected waste into organic and inorganic categories.
- Perform real-time detection using a webcam.
- Integrate the trained model into a Flask web application.
- Explore the application of computer vision to support waste sorting.

## 🛠️ Tech Stack

- **Programming Language:** Python, HTML, CSS
- **Computer Vision:** OpenCV
- **Object Detection:** YOLOv8, Ultralytics
- **Dataset Annotation:** Roboflow
- **Web Framework:** Flask
- **Model Training:** Ultralytics YOLO

## ✨ Features

- **Waste Detection:** Detects organic and inorganic waste objects.
- **Real-Time Webcam:** Processes camera frames for object detection.
- **Region of Interest (ROI):** Focuses detection on a predefined area of the camera frame.
- **Confidence Threshold:** Filters detections based on a configurable confidence threshold.
- **Web Interface:** Provides login, navigation, and detection pages.
- **Detection Controls:** Allows users to start and stop detection.

## 🗂️ Dataset
#### Use This Code on Notebook
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="dqCpF4UkgspXf7kIVql3")
project = rf.workspace("blasss-workspace").project("github-a2mu4")

version = project.version(1)

dataset = version.download("yolov8")
                
## 🔮 Future Improvements

- Improve model accuracy and generalization.
- Evaluate performance under different lighting and background conditions.
- Improve the web interface and camera controls.
- Explore deployment options for practical waste sorting applications.

## Result
### Web Application
<img width="700" height="310" alt="image" src="https://github.com/user-attachments/assets/50c9d5aa-e488-4732-a4a3-da6a39f4b310" />

### Object Detection Results
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/37b43f2c-e4ac-40a8-8235-a7b7d0528136" />

## 👤 Author

**Muhammad Rifqi Sahib**

GitHub: B1ass

---

*This project was developed for learning and experimentation in computer vision, object detection, and web application integration.*
