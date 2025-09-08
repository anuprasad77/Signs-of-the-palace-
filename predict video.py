import cv2
import numpy as np
import tensorflow as tf
import pickle
import os

# === INPUT: Full path to video ===
VIDEO_PATH = r"C:\Users\Admin\Desktop\palace\test\ambulance carriage\01.mp4"

# === SETTINGS ===
NUM_FRAMES = 8
IMG_HEIGHT, IMG_WIDTH = 112, 112

# === Load Model & Class Names ===
model = tf.keras.models.load_model("sign_language_model_augmented.h5")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# === Extract 16 evenly spaced frames ===
def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames or total_frames == 0:
        print("❌ Not enough frames in video!")
        cap.release()
        return None

    idxs = np.linspace(0, total_frames-1, num_frames).astype(int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frames.append(frame)

    cap.release()
    if len(frames) != num_frames:
        return None

    frames = np.array(frames).astype(np.float32) / 255.0
    return np.expand_dims(frames, axis=0)  # (1, 16, 224, 224, 3)

# === Prediction ===
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video not found at: {VIDEO_PATH}")
else:
    frames = extract_frames(VIDEO_PATH)
    if frames is not None:
        pred = model.predict(frames)
        top_index = np.argmax(pred[0])
        label = class_names[top_index]
        confidence = pred[0][top_index]
        print(f"\n🔮 Prediction: {label} ({confidence:.2f} confidence)")
    else:
        print("❌ Frame extraction failed.")
