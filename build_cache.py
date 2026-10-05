import os
import time
import pickle
import face_recognition

faces_dir = "faces"
CACHE_FILE = "encodings_cache.pkl"

print(f"[*] Pre-computing face encodings from '{faces_dir}'...")
start_t = time.time()
encodings = []
names = []

for filename in sorted(os.listdir(faces_dir)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filepath = os.path.join(faces_dir, filename)
        img = face_recognition.load_image_file(filepath)
        encs = face_recognition.face_encodings(img)
        if len(encs) > 0:
            encodings.append(encs[0])
            name = os.path.splitext(filename)[0]
            names.append(name)
            print(f"  [+] Encoded: {name}")
        else:
            print(f"  [-] Warning: No face in {filename}")

with open(CACHE_FILE, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print(f"[*] Successfully saved {len(names)} encodings to '{CACHE_FILE}' in {time.time() - start_t:.2f}s!")
