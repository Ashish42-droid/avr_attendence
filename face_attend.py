import cv2
import face_recognition
import os
import time
import pickle
import numpy as np
from attendance_manager import AttendanceManager

print("Initializing Attendance System...")
attendance_mgr = AttendanceManager(
    attendance_dir="attendance",
    cooldown_seconds=60,               # Ignore repeated punches within 1 min
    out_time_threshold_seconds=120     # Record Out-Time if scanned after 2 min
)

# Load Reference Faces with Cache for Instant Startup
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
    print(f"Loaded {len(known_face_names)} faces instantly from cache!")
else:
    print(f"Extracting encodings from '{faces_dir}' directory (Run build_cache.py to speed up next time)...")
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
        # Save cache for next run
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
    else:
        print(f"Warning: '{faces_dir}' directory not found.")

print(f"Finished loading {len(known_face_names)} known faces.")

import numpy as np

def open_working_camera():
    """
    Scans camera devices and automatically selects the first one delivering
    a real, non-black video feed (avoids Windows dummy/IR camera black screens).
    """
    print("[*] Probing for active camera...")
    # Prioritize index 1 (physical webcam), then 0, then 2
    for idx in [1, 0, 2]:
        for backend_name, backend in [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Warm up and check if camera actually delivers non-black frames
                working = False
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and np.mean(frame) > 5:
                        working = True
                        break
                if working:
                    print(f"[+] Successfully connected to Camera Index {idx} using {backend_name}!")
                    return cap
                cap.release()
    return None

video_capture = open_working_camera()
if video_capture is None:
    print("[ERROR] Could not find any active webcam delivering video.")
    print("[*] Tip: Check if another application (Zoom, Teams, Browser) has the camera locked.")
    exit(1)

print("Webcam started successfully! Press 'q' on your keyboard to exit.")

frame_count = 0
face_locations = []
face_names = []
process_every_n_frames = 2  # Process every 2nd frame for maximum FPS
fps_time = time.time()
fps = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    # Calculate real-time FPS
    if frame_count % 10 == 0:
        fps = round(10 / (time.time() - fps_time), 1)
        fps_time = time.time()

    # Only process every N-th frame to save CPU
    if frame_count % process_every_n_frames == 0:
        # Resize frame to 1/4 size for 16x faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations & encodings on the smaller frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = face_distances.argmin()
                # 0.55 distance threshold for high accuracy
                if matches[best_match_index] and face_distances[best_match_index] < 0.55:
                    name = known_face_names[best_match_index]
                    # Automate attendance recording in Excel
                    attendance_mgr.mark_attendance(name)

            face_names.append(name)

    # Display results on the full-sized original frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since we detected them on 1/4 size frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw box around face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Draw label box below face
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw live attendance confirmation banner on frame
    frame = attendance_mgr.draw_banner_overlay(frame)

    # Show FPS in bottom-left
    cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the result
    cv2.imshow('Face Recognition Attendance System', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()