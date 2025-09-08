import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from descriptions import descriptions  # External class descriptions

# === CONFIGURATION ===
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = {'mp4'}
NUM_FRAMES = 8
IMG_SIZE = 112

# === INITIALIZE FLASK APP ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === LOAD MODEL AND CLASS NAMES ===
model = tf.keras.models.load_model("sign_language_model_augmented.h5")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# === UTILITY FUNCTIONS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < NUM_FRAMES:
        return None

    idxs = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
    frames = []
    i = 0
    success = True

    while success and len(frames) < NUM_FRAMES:
        success, frame = cap.read()
        if i in idxs:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        i += 1

    cap.release()

    if len(frames) < NUM_FRAMES:
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0
    return np.expand_dims(frames, axis=0)

# === ROUTES ===

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "video" not in request.files:
            return render_template("predict.html", error="No file uploaded.")

        file = request.files["video"]
        if file.filename == "":
            return render_template("predict.html", error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(STATIC_FOLDER, exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            frames = extract_frames(filepath)
            if frames is None:
                return render_template("predict.html", error="Video must have at least 8 frames.")

            pred = model.predict(frames)
            class_idx = np.argmax(pred)
            predicted_class = class_names[class_idx]

            # Description from separate file
            class_key = predicted_class.lower().strip()
            description = descriptions.get(class_key, "No description available.")

            # Save middle frame
            middle_frame = frames[0][NUM_FRAMES // 2] * 255.0  # Unnormalize
            image_filename = filename.rsplit('.', 1)[0] + "_frame.jpg"
            image_path = os.path.join(STATIC_FOLDER, image_filename)
            cv2.imwrite(image_path, middle_frame.astype(np.uint8))

            return render_template("predict.html", prediction=predicted_class, description=description, image_file=image_filename)

    return render_template("predict.html")

@app.route("/about-palace")
def about_palace():
    return render_template("about_palace.html")

@app.route("/about-project")
def about_project():
    return render_template("about_project.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)
