# 🛡️ Invicti Sign2Sound: Real-Time ASL Translation Engine

## 📖 Project Overview
Sign2Sound is a lightweight, real-time American Sign Language (ASL) alphabet recognition system built by Team Invicti. Using advanced computer vision and a custom Deep Neural Network (DNN), the system translates live webcam gestures into readable text and synthesized speech. 

By utilizing Google MediaPipe for skeletal extraction and applying strict translation-invariant mathematics, this engine bypasses the heavy processing requirements of traditional Convolutional Neural Networks (CNNs), allowing it to run flawlessly in real-time on standard CPU hardware.

*Note: The current implementation focuses on static ASL alphabet letters and excludes dynamic (motion-based) gestures like 'J' and 'Z'.*

---

## ✨ Key Features
* **Real-Time Translation:** Captures and classifies hand gestures at 30 FPS with < 1-second latency.
* **Translation Invariance:** Utilizes wrist-relative 3D coordinate math, meaning the AI recognizes the hand shape regardless of where it is positioned on the screen.
* **Audio Accessibility:** Integrates an offline Text-to-Speech (TTS) engine running on a background daemon thread, ensuring the system speaks translated words without freezing the live video feed.
* **Modern GUI:** Features a sleek, responsive dark-mode interface built with CustomTkinter.
* **Temporal Smoothing:** Implements a rolling stability buffer across consecutive frames to prevent screen flickering and guarantee confident predictions.

---

## 🧠 System Architecture & Data Flow
Our pipeline completely isolates the geometry of the hand from environmental noise (like background clutter or lighting changes):

1. **Vision Pipeline:** OpenCV captures the live RGB feed and passes it to Google MediaPipe.
2. **Feature Extraction:** MediaPipe identifies 21 3D hand landmarks (63 total x, y, z coordinates).
3. **Mathematical Normalization:** The 63 coordinates are converted to "wrist-relative" values (subtracting the wrist's position from all fingers) and normalized to a -1 to 1 scale. 
4. **Classification:** A custom 4-layer Feed-Forward Dense Neural Network (DNN) with Batch Normalization and Dropout layers processes the 1D coordinate array and outputs a probability vector for the 24 static alphabet classes.
5. **Output:** The predicted character is bridged to the CustomTkinter UI via Pillow (PIL) and spoken via pyttsx3.

---

## 📊 Dataset & Augmentation
The model was trained on a custom dataset of ASL spatial coordinates. To prevent overfitting and drastically improve real-world generalization, we applied **Synthetic Jitter Augmentation**. 

By injecting 1.5% Gaussian noise into the 1D coordinate arrays during training, we simulated natural human hand-shaking, expanding our dataset by 400% and stabilizing the model's accuracy to 97%.

---

## 🛠️ Requirements and Dependencies
This project was developed using **Python 3.11**. All necessary library versions (including OpenCV, MediaPipe, and TensorFlow) are strictly defined to ensure a stable build.

To install everything instantly, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

### Step 1: Preprocess the Data
*(Extracts MediaPipe landmarks from raw training images and saves them as `.npy` coordinate arrays)*
```bash
python preprocess.py
```
### Step 2: Train the Neural Network
```bash
python train.py
```
### Step 3: Launch the Sign2Sound UI
```bash
python gui_app.py
```

