import cv2
import numpy as np
import os
import time
import multiprocessing
import pyttsx3
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from collections import deque
import customtkinter as ctk
from PIL import Image, ImageTk

# ==================================================
# 1. THE PERSISTENT WORKER (Refreshes engine every time)
# ==================================================
def tts_worker(conn):
    """
    Stays alive but recreates the engine for every request.
    This prevents the 'one-time-only' bug common on Windows drivers.
    """
    while True:
        try:
            text = conn.recv() 
            if text == "STOP":
                break
            
            # Re-initializing INSIDE the loop ensures a fresh driver state
            engine = pyttsx3.init()
            engine.setProperty('rate', 200) 
            engine.say(text)
            engine.runAndWait()
            
            # Explicitly stop and delete the engine instance to free the driver
            engine.stop()
            del engine
            
        except EOFError:
            break
        except Exception as e:
            print(f"TTS Worker Error: {e}")

# ==================================================
# 2. UI APPLICATION CLASS
# ==================================================
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignToSound - Final Stable Version")
        self.root.geometry("1000x650")
        self.root.resizable(False, False)

        # --- SETUP MULTIPROCESSING PIPE ---
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(target=tts_worker, args=(self.child_conn,), daemon=True)
        self.proc.start()

        # Load AI Model & MediaPipe
        # Ensure 'sign_language_model.keras' and 'hand_landmarker.task' are in the script folder
        self.model = load_model("sign_language_model.keras")
        self.DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")
        self.actions = np.array(sorted([
            d for d in os.listdir(self.DATA_PATH)
            if os.path.isdir(os.path.join(self.DATA_PATH, d)) and len(d) == 1 and d.isalpha()
        ]))

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Logic Variables
        self.prediction_buffer = deque(maxlen=5)
        self.current_stable_letter = ""
        self.stable_frames = 0
        self.CONFIDENCE_THRESHOLD = 0.70  
        self.REQUIRED_FRAMES = 5          
        self.letter_buffer = []
        self.word = ""  
        self.last_seen_time = time.time()
        self.PAUSE_TIME = 1.5 
        
        self.auto_speak_var = ctk.BooleanVar(value=False) 
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.build_ui()
        self.update_video()

    def speak_word(self, text):
        """Sends the text to the background process instantly."""
        if text.strip():
            try:
                self.parent_conn.send(text)
            except:
                pass

    def build_ui(self):
        # UI Setup (Video and Info Frames)
        self.video_frame = ctk.CTkFrame(self.root, width=660, height=500, corner_radius=10)
        self.video_frame.place(x=20, y=20)
        self.video_label = ctk.CTkLabel(self.video_frame, text="Webcam Loading...")
        self.video_label.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        self.info_frame = ctk.CTkFrame(self.root, width=280, height=500, corner_radius=10)
        self.info_frame.place(x=700, y=20)

        ctk.CTkLabel(self.info_frame, text="Current Letter", font=("Arial", 20, "bold")).place(x=20, y=20)
        self.letter_display = ctk.CTkLabel(self.info_frame, text="-", font=("Arial", 80, "bold"), text_color="#00FFCC")
        self.letter_display.place(relx=0.5, y=100, anchor=ctk.CENTER)

        self.conf_display = ctk.CTkLabel(self.info_frame, text="Confidence: 0%", font=("Arial", 14))
        self.conf_display.place(x=20, y=170)

        self.stable_display = ctk.CTkLabel(self.info_frame, text="Stability: 0/5", font=("Arial", 14))
        self.stable_display.place(x=20, y=200)

        self.bottom_frame = ctk.CTkFrame(self.root, width=960, height=80, corner_radius=10)
        self.bottom_frame.place(x=20, y=540)
        ctk.CTkLabel(self.bottom_frame, text="Sentence:", font=("Arial", 18, "bold")).place(x=20, y=25)
        self.word_display = ctk.CTkLabel(self.bottom_frame, text="", font=("Arial", 28, "bold"), text_color="#FFD700")
        self.word_display.place(x=130, y=22)

        # Controls
        ctk.CTkButton(self.info_frame, text="␣ Space", command=self.add_space, fg_color="#3498DB").place(relx=0.5, y=270, anchor=ctk.CENTER)
        ctk.CTkButton(self.info_frame, text="⌫ Delete", command=self.delete_last, fg_color="#F39C12").place(relx=0.5, y=310, anchor=ctk.CENTER)
        ctk.CTkButton(self.info_frame, text="🗑️ Clear", command=self.clear_word, fg_color="#E74C3C").place(relx=0.5, y=350, anchor=ctk.CENTER)
        ctk.CTkButton(self.info_frame, text="🔊 Speak", command=self.manual_speak, fg_color="#2ECC71").place(relx=0.5, y=400, anchor=ctk.CENTER)
        
        self.auto_speak_switch = ctk.CTkSwitch(self.info_frame, text="Auto-Speak", variable=self.auto_speak_var)
        self.auto_speak_switch.place(relx=0.2, y=450)

    def add_space(self):
        if not self.word.endswith(" "):
            if self.auto_speak_var.get() and self.word.strip():
                self.speak_word(self.word.split()[-1])
            self.word += " "
            self.word_display.configure(text=self.word)
            self.letter_buffer = []

    def delete_last(self):
        if self.word:
            self.word = self.word[:-1]
            self.word_display.configure(text=self.word)
            self.letter_buffer = []

    def clear_word(self):
        self.word = ""
        self.letter_buffer = []
        self.word_display.configure(text="")

    def manual_speak(self):
        self.speak_word(self.word)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                wrist_x, wrist_y, wrist_z = hand[0].x, hand[0].y, hand[0].z
                landmarks = []
                for lm in hand:
                    landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                
                landmarks = np.array(landmarks, dtype=np.float32)
                max_val = np.max(np.abs(landmarks))
                if max_val > 0: landmarks /= max_val
                
                landmarks = landmarks.reshape(1, 63)
                prediction = self.model.predict(landmarks, verbose=0)
                self.prediction_buffer.append(prediction)

                avg_pred = np.mean(self.prediction_buffer, axis=0)
                class_id = np.argmax(avg_pred)
                confidence = np.max(avg_pred)
                letter = self.actions[class_id]
                
                self.conf_display.configure(text=f"Confidence: {int(confidence*100)}%")

                if confidence > self.CONFIDENCE_THRESHOLD:
                    if letter == self.current_stable_letter:
                        self.stable_frames += 1
                    else:
                        self.current_stable_letter = letter
                        self.stable_frames = 1

                    self.stable_display.configure(text=f"Stability: {self.stable_frames}/5")

                    if self.stable_frames == self.REQUIRED_FRAMES:
                        self.letter_display.configure(text=letter)
                        if len(self.letter_buffer) == 0 or letter != self.letter_buffer[-1]:
                            self.letter_buffer.append(letter)
                            self.word += letter  
                            self.word_display.configure(text=self.word) 
                            self.last_seen_time = time.time()
                else:
                    self.stable_frames = 0
            else:
                self.stable_frames = 0
                self.current_stable_letter = ""
                if len(self.letter_buffer) > 0 and (time.time() - self.last_seen_time) > self.PAUSE_TIME:
                    self.letter_buffer = [] 

            img = Image.fromarray(rgb)
            img = img.resize((640, 480))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(text="", image=imgtk)

        self.root.after(10, self.update_video)

    def on_close(self):
        try:
            self.parent_conn.send("STOP")
        except:
            pass
        self.cap.release()
        self.root.destroy()

# ==================================================
# 3. RUN APP
# ==================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_root = ctk.CTk()
    app = SignLanguageApp(app_root)
    app_root.protocol("WM_DELETE_WINDOW", app.on_close)
    app_root.mainloop()