import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 1. LOAD DATA
# ==================================================
DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")

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
# 2. NORMALIZATION
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
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ==================================================
# 6. SAVE MODEL
# ==================================================
model.save("sign_language_model.keras")
print("✅ Model saved as sign_language_model.keras")

# ==================================================
# 7. TEST SET PERFORMANCE
# ==================================================
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report (Test Set):")
print(classification_report(y_true, y_pred_labels, target_names=actions))

# ==================================================
# 8. CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_true, y_pred_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=actions,
    yticklabels=actions,
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==================================================
# 9. TRAINING CURVES
# ==================================================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# ==================================================
# 10. FINAL TEST ACCURACY
# ==================================================
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Final Test Accuracy: {acc * 100:.2f}%")
