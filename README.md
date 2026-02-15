<<<<<<< HEAD


=======
>>>>>>> a0d53ec (Clean initial commit)
# Real-Time American Sign Language (ASL) Alphabet Recognition

## Project Overview

This project implements a real-time American Sign Language (ASL) alphabet recognition system using computer vision and deep learning techniques. The system recognizes individual ASL letters (A–Z) from live webcam input and converts them into readable text, with optional speech output.

**Note:** The current implementation supports **letter-level recognition only** and does not perform word or sentence translation.

---

## Objectives

The primary objectives of this project are to detect and recognize ASL alphabet gestures (A–Z) in real time, provide stable and accurate predictions using hand landmark features, and enable accessible communication through textual and optional speech output.

---

## System Architecture

The system follows a structured end-to-end pipeline. Live video input is captured from a webcam and each frame is converted to RGB format before being processed by MediaPipe for hand landmark extraction. A total of 21 hand landmarks are detected, each represented by x, y, and z coordinates, resulting in a 63-dimensional feature vector. These features are passed to a trained neural network model for ASL alphabet classification. To ensure prediction stability, temporal smoothing is applied across consecutive frames. The final output is displayed as recognized text on the screen, along with a confidence score and optional speech feedback.

---

## Dataset Used

The model is trained using augmented ASL RGB hand gesture images sourced from **Train Data 1** and **Train Data 2**. These images are processed using MediaPipe to extract hand landmarks. Skeleton data and other pre-processed image variants are excluded, as the system relies on landmark extraction from raw RGB images for optimal performance and accuracy.

---

## Requirements and Dependencies

The system requires Python 3.11 along with the following libraries and components: OpenCV, MediaPipe, TensorFlow/Keras, NumPy, and a webcam for real-time testing.

---

## How to Run the Project

### Step 1: Preprocess the Dataset

```bash
python preprocess.py
```

### Step 2: Train the Model

```bash
python train.py
```

### Step 3: Run Real-Time Testing

```bash
python realtime_test.py
```

Press **`q`** to exit the webcam window.

---

## Model Capabilities and Limitations

### Capabilities

The system supports real-time ASL alphabet recognition (A–Z), provides stable predictions using temporal smoothing, and operates efficiently on standard computing hardware.

### Limitations

The current implementation does not support word or sentence-level sign recognition, recognizes only single-hand gestures, and may be affected by variations in lighting conditions and hand visibility.

---

## Future Improvements

Future enhancements may include extending the system to support word and sentence-level recognition, enabling multi-hand and dynamic gesture support, deploying the system as a mobile or web-based application, and improving accuracy through larger and more diverse datasets.


