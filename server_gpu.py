import os
import time
import pickle
import numpy as np
import cv2
import torch
import face_recognition
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from attendance_manager import AttendanceManager

app = FastAPI(title="DGX GPU Facial Recognition Server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "encodings_cache.pkl")
faces_dir = os.path.join(BASE_DIR, "faces")

# Initialize Attendance Manager
attendance_mgr = AttendanceManager(
    attendance_dir=os.path.join(BASE_DIR, "attendance"),
    cooldown_seconds=60,
    out_time_threshold_seconds=120
)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Compute Device: {device.upper()}")
if device == "cuda":
    print(f"[*] GPU Model     : {torch.cuda.get_device_name(0)}")

# Load Reference Faces with Cache
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    if os.path.exists(CACHE_FILE):
        print(f"[*] Loading cached face encodings from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
        print(f"[*] Loaded {len(known_face_names)} faces from cache.")
    else:
        print(f"[*] Computing face encodings from '{faces_dir}'...")
        if os.path.exists(faces_dir):
            for filename in sorted(os.listdir(faces_dir)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    filepath = os.path.join(faces_dir, filename)
                    img = face_recognition.load_image_file(filepath)
                    encs = face_recognition.face_encodings(img)
                    if len(encs) > 0:
                        known_face_encodings.append(encs[0])
                        name = os.path.splitext(filename)[0]
                        known_face_names.append(name)
                        print(f"  [+] Loaded: {name}")
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
            print(f"[*] Saved {len(known_face_names)} encodings to '{CACHE_FILE}'.")
        else:
            print(f"[!] Warning: '{faces_dir}' not found.")

load_known_faces()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "faces_loaded": len(known_face_names)
    }

@app.post("/recognize")
async def recognize_frame(file: UploadFile = File(...)):
    start_t = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Failed to decode frame"}, status_code=400)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        min_dist = 1.0
        punch_res = None

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_idx = np.argmin(face_distances)
            min_dist = float(face_distances[best_match_idx])
            
            # Threshold for recognition match
            if min_dist < 0.55:
                name = known_face_names[best_match_idx]
                # Automatically record attendance in Excel
                punch_res = attendance_mgr.mark_attendance(name)

        results.append({
            "name": name,
            "box": [int(top), int(right), int(bottom), int(left)],
            "distance": round(min_dist, 4),
            "attendance": punch_res
        })

    inference_ms = round((time.time() - start_t) * 1000, 2)
    return {
        "faces": results,
        "inference_ms": inference_ms
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
