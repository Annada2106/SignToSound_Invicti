import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import mediapipe as mp

# ==================================================
# 1. LOAD MODEL & CLASSES
# ==================================================
model = load_model("sign_language_model.keras")

DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")
actions = np.array(sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and len(d) == 1 and d.isalpha()
]))

print("✅ Model loaded")
print("Classes:", actions)

# ==================================================
# 2. MEDIAPIPE HAND LANDMARKER
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ==================================================
# 3. PREDICTION SMOOTHING
# ==================================================
prediction_buffer = deque(maxlen=15)

# ==================================================
# 4. WEBCAM (WINDOWS FIX)
# ==================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible. Check camera permissions or index.")

print("🎥 Webcam started. Press 'q' to quit.")

# ==================================================
# 5. LOOP
# ==================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32
        ).flatten()

        # SAME normalization as training
        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val

        landmarks = landmarks.reshape(1, 63)

        prediction = model.predict(landmarks, verbose=0)
        prediction_buffer.append(prediction)

        avg_pred = np.mean(prediction_buffer, axis=0)
        class_id = np.argmax(avg_pred)
        confidence = np.max(avg_pred)

        letter = actions[class_id]

        # Display
        cv2.rectangle(frame, (0, 0), (320, 90), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"LETTER: {letter}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3
        )
        cv2.putText(
            frame,
            f"CONF: {confidence:.2f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "SHOW ONE HAND",
            (60, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
