import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==================================================
# 1. LOAD DATA
# ==================================================
DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")

# Only A–Z folders
actions = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and len(d) == 1 and d.isalpha()
])

actions = np.array(actions)
print("Classes:", actions)

X, y = [], []
label_map = {label: idx for idx, label in enumerate(actions)}

for letter in actions:
    letter_path = os.path.join(DATA_PATH, letter)
    for file in os.listdir(letter_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(letter_path, file))
            if data.shape == (63,):
                X.append(data)
                y.append(label_map[letter])

X = np.array(X, dtype=np.float32)
y = to_categorical(y, num_classes=len(actions))

if X.shape[0] == 0:
    raise ValueError("❌ No training samples found. Run preprocess.py first.")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==================================================
# 2. NORMALIZATION (SAME AS TESTING)
# ==================================================
max_vals = np.max(np.abs(X), axis=1)
max_vals[max_vals == 0] = 1
X = X / max_vals[:, np.newaxis]

# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==================================================
# 4. MODEL
# ==================================================
model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(len(actions), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==================================================
# 5. TRAIN
# ==================================================
print("🚀 Training started...")
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ==================================================
# 6. SAVE
# ==================================================
model.save("sign_language_model.keras")
print("✅ Model saved as sign_language_model.keras")

# ==================================================
# 7. EVALUATE
# ==================================================
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {acc * 100:.2f}%")
