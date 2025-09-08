import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle

# === PARAMETERS ===
DATA_DIR = r"C:\Users\Admin\Desktop\palace\train_augmented"
NUM_CLASSES = 29
NUM_FRAMES = 8
IMG_HEIGHT, IMG_WIDTH = 112, 112
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 15

# === FRAME EXTRACTION ===
def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[WARNING] No frames found in: {video_path}")
        cap.release()
        return None

    frame_idxs = np.linspace(0, total_frames-1, num_frames).astype(int)
    idx_set = set(frame_idxs)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frames.append(frame)

    cap.release()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    if len(frames) != num_frames:
        print(f"[ERROR] Not enough frames after extraction: {video_path}")
        return None

    return np.array(frames, dtype=np.float16) / 255.0

# === GENERATOR ===
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.video_paths[k] for k in indices]
        batch_labels = [self.labels[k] for k in indices]

        X = []
        for path in batch_paths:
            frames = extract_frames(path)
            if frames is not None:
                X.append(frames)
            else:
                # Add a zero tensor if extraction failed (prevents crashing)
                X.append(np.zeros((NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float16))

        return np.array(X, dtype=np.float16), np.array(batch_labels, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# === DATA PREPARATION ===
def prepare_dataset(data_dir):
    video_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_folder = os.path.join(data_dir, cls_name)
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for file in files:
            video_paths.append(os.path.join(cls_folder, file))
            labels.append(class_to_idx[cls_name])

    print(f"✅ Total videos found: {len(video_paths)}")
    return video_paths, labels, class_names

# === MODEL BUILDING ===
def build_model(num_classes):
    input_layer = layers.Input(shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = layers.TimeDistributed(base_model)(input_layer)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    return model

# === MAIN FUNCTION ===
def main():
    video_paths, labels, class_names = prepare_dataset(DATA_DIR)
    if len(video_paths) == 0:
        print("❌ No videos found. Check the dataset path.")
        return

    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels)

    train_gen = VideoDataGenerator(X_train_paths, y_train)
    val_gen = VideoDataGenerator(X_val_paths, y_val)

    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    print(f"\n📊 Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    model.save("sign_language_model_augmented.h5")
    print("✅ Model saved as sign_language_model_augmented.h5")

    with open("class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)
    print("✅ Class names saved to class_names.pkl")

if __name__ == "__main__":
    main()
