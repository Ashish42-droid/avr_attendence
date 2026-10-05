import cv2
import face_recognition
import os
import time
import pickle
import numpy as np
import torch
from attendance_manager import AttendanceManager

print("=" * 60)
print("FACIAL RECOGNITION ATTENDANCE SERVER (DGX GPU)")
print("=" * 60)

# Initialize Attendance Manager
attendance_mgr = AttendanceManager(
    attendance_dir="attendance",
    cooldown_seconds=60,
    out_time_threshold_seconds=120
)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Compute Device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cache encodings to avoid re-encoding on every run
CACHE_FILE = "encodings_cache.pkl"
known_face_encodings = []
known_face_names = []

faces_dir = "faces"

if os.path.exists(CACHE_FILE):
    print(f"Loading cached face encodings from '{CACHE_FILE}'...")
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} cached faces instantly.")
else:
    print(f"Extracting encodings from '{faces_dir}' directory...")
    if os.path.exists(faces_dir):
        for filename in sorted(os.listdir(faces_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"  [+] Loaded: {name}")
                else:
                    print(f"  [-] Warning: No face found in {filename}")
        # Save cache
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        print(f"Saved {len(known_face_names)} encodings to '{CACHE_FILE}'.")
    else:
        print(f"Warning: '{faces_dir}' directory not found.")

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.1.18:554/stream1")
print(f"\nConnecting to video source: {RTSP_URL}")
video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not video_capture.isOpened():
    print(f"Warning: Could not connect to {RTSP_URL}. Testing with local images/mode.")
else:
    print("Stream connected successfully! Starting recognition loop...")

frame_count = 0
start_time = time.time()

try:
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Stream ended or frame drop...")
            time.sleep(0.1)
            continue

        frame_count += 1
        # Process every 2nd frame for maximum throughput
        if frame_count % 2 != 0:
            continue

        # Resize frame for ultra fast processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations & encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    dist = face_distances[best_match_index]
                    # Automate attendance recording in Excel
                    attendance_mgr.mark_attendance(name)

        # Headless display check
        if "DISPLAY" in os.environ:
            frame = attendance_mgr.draw_banner_overlay(frame)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping attendance server...")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    print("Cleaned up video resources.")
