import cv2
import os

input_root = "train_augmented"
output_root = "train_images"
frames_per_video = 10  # Reduce if needed for speed

for class_name in os.listdir(input_root):
    class_folder = os.path.join(input_root, class_name)
    output_class_folder = os.path.join(output_root, class_name)
    os.makedirs(output_class_folder, exist_ok=True)

    for file_name in os.listdir(class_folder):
        if not file_name.endswith('.mp4'):
            continue

        video_path = os.path.join(class_folder, file_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // frames_per_video)
        frame_idx = 0
        saved = 0

        while saved < frames_per_video and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            output_path = os.path.join(
                output_class_folder,
                f"{os.path.splitext(file_name)[0]}_frame{saved}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved += 1
            frame_idx += step

        cap.release()
        print(f"✔ Extracted {saved} frames from {file_name}")
