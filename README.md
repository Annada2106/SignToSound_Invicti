Real-Time American Sign Language (ASL) Alphabet Recognition
 Project Overview

This project implements a real-time American Sign Language (ASL) alphabet recognition system using computer vision and deep learning. The system recognizes individual ASL letters (A–Z) from live webcam input and converts them into readable text, with optional speech output.
Note: The current implementation supports letter-level recognition only and does not perform word or sentence translation.

Objectives
Detect and recognize ASL alphabet gestures (A–Z) in real time
Provide stable and accurate predictions using hand landmark features
Enable accessible communication through text and optional speech output

 System Architecture
Input Capture: Live video stream from a webcam
Preprocessing: RGB conversion and hand landmark extraction using MediaPipe
Feature Representation: 21 hand landmarks (x, y, z → 63 features)
Model Inference: Neural network-based classification (A–Z)
Post-processing: Temporal smoothing for prediction stability
Output: Display of recognized letter with confidence score and optional speech

Dataset Used
The model is trained using augmented ASL RGB hand gesture images sourced from Train Data 1 and Train Data 2. These images are processed using MediaPipe to extract hand landmarks. Skeleton and pre-processed image variants are excluded, as the system relies on landmark extraction from raw RGB images.

Requirements & Dependencies
Python 3.11
OpenCV
MediaPipe
TensorFlow / Keras
NumPy
Webcam (for real-time testing)

How to Run the Project
1.Preprocess the Dataset:
python preprocess.py
2.Train the Model:
python train.py
3.Run Real-Time Testing:
python realtime_test.py
Press q to exit the webcam window.

Model Capabilities & Limitations
Capabilities:
  Real-time ASL alphabet recognition (A–Z)
  Stable predictions with temporal smoothing
  Works efficiently on standard hardware
Limitations:
  Word and sentence-level sign recognition
  Multi-hand and dynamic gesture support
  Mobile and web-based deployment
  Improved accuracy through larger datasets


