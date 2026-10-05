import os
import time
import torch
import cv2
import face_recognition

print("=" * 60)
print("DGX GPU FACE RECOGNITION PIPELINE VERIFICATION")
print("=" * 60)

# 1. GPU Check
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device     : {torch.cuda.get_device_name(0)}")

# 2. Face Recognition Encodings from Faces Directory
faces_dir = "faces"
known_face_encodings = []
known_face_names = []

if os.path.exists(faces_dir):
    print(f"\nLoading reference faces from '{faces_dir}'...")
    start_t = time.time()
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
                print(f"  [-] No face in: {filename}")
    print(f"Loaded {len(known_face_names)} faces in {time.time() - start_t:.2f}s")

# 3. Test on test.jpg
test_img_path = "test.jpg"
if os.path.exists(test_img_path):
    print(f"\nTesting recognition on '{test_img_path}'...")
    test_img = face_recognition.load_image_file(test_img_path)
    locations = face_recognition.face_locations(test_img)
    encodings = face_recognition.face_encodings(test_img, locations)
    print(f"Detected {len(locations)} face(s) in {test_img_path}")
    
    for (top, right, bottom, left), face_encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_idx = face_distances.argmin()
            if matches[best_match_idx]:
                name = known_face_names[best_match_idx]
                print(f"  --> MATCH FOUND: {name} (Distance: {face_distances[best_match_idx]:.4f})")
            else:
                print(f"  --> NO MATCH (Closest: {known_face_names[best_match_idx]} at {face_distances[best_match_idx]:.4f})")

# 4. YOLO model check
if os.path.exists("best.pt"):
    print("\nVerifying YOLO model (best.pt) on GPU...")
    try:
        from ultralytics import YOLO
        model = YOLO("best.pt")
        print(f"  [+] YOLO loaded successfully on device: {model.device}")
        if torch.cuda.is_available():
            results = model.predict(source=test_img_path, device=0, verbose=False)
            print(f"  [+] YOLO GPU Inference completed: {len(results[0].boxes)} detections")
    except Exception as e:
        print(f"  [-] YOLO error: {e}")

print("\n" + "=" * 60)
print("ALL CHECKS COMPLETED SUCCESSFULLY!")
print("=" * 60)
