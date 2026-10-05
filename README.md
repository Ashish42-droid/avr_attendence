# 🚀 AVR Face Recognition Attendance System

An automated, real-time facial recognition attendance system featuring smart daily Excel logging, dual-punch tracking (In-Time and Out-Time with duration calculation), anti-duplicate cooldown protection, and DGX GPU client-server streaming.

---

## ✨ Features

- **Real-Time Facial Recognition**: Powered by `face_recognition` (dlib) with 16x accelerated quarter-scale processing.
- **Automated Daily Excel Logging**: Automatically generates and updates daily formatted spreadsheets (`attendance/Attendance_YYYY-MM-DD.xlsx`).
- **Dual-Punch Logic (In & Out)**:
  - **First Scan**: Records `In-Time` and sets status to `Present`.
  - **Cooldown Filter**: Discards duplicate scans within 60 seconds.
  - **Subsequent Scan**: Updates `Out-Time` and recalculates total duration worked (e.g. `08h 15m`).
- **Lock-Safe Protection (`PermissionError` Immunity)**: If someone has the `.xlsx` file open in Microsoft Excel, punches are saved to an emergency CSV backup without crashing the video stream.
- **Smart Camera Hardware Prober**: Automatically detects and binds to the real physical webcam on Windows (DirectShow), avoiding dummy/black IR camera streams.
- **Live HUD Banner Overlay**: Displays semi-transparent status banners directly on the camera preview.
- **Dual Architecture**:
  - **Standalone Mode**: Single script (`face_attend.py`) for local webcams or RTSP feeds.
  - **Client-Server Mode**: High-throughput FastAPI DGX GPU server (`server_gpu.py`) with lightweight edge camera clients (`client_camera.py`).

---

## 📁 Repository Structure

```text
avr_attendance/
├── attendance/              # Generated daily Excel attendance reports
├── faces/                   # Reference images of known individuals (e.g. John_Doe.jpg)
├── attendance_manager.py    # Core Excel automation & punch logic
├── face_attend.py           # Standalone live face recognition & attendance script
├── build_cache.py           # Pre-computes face encodings for instant startup
├── server_gpu.py            # FastAPI GPU acceleration server
├── client_camera.py         # Live camera stream client with HUD banner
├── face_attend_server.py    # Direct RTSP video stream monitor
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ashish42-droid/avr_attendence.git
cd avr_attendence
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Add Known Faces
Place clear, front-facing photos in the `faces/` directory. Name each file with the person's name (e.g., `Vishal.jpg`, `Abhinav.png`).

### Step 2: Build the Fast Encodings Cache
```bash
python build_cache.py
```
*Generates `encodings_cache.pkl` so recognition starts in 0.05s instead of recalculating raw images every run.*

---

### Step 3: Launch Attendance

#### Option A: Standalone Mode (RTSP IP Camera or Local Webcam)
```bash
# Connects automatically to RTSP IP Camera (rtsp://192.168.1.18:554/stream1)
python face_attend.py

# Or explicitly pass RTSP stream / webcam index:
python face_attend.py --source "rtsp://192.168.1.18:554/stream1"
python face_attend.py --source 0
```
*Press `q` in the video window to exit.*

#### Option B: DGX GPU Client-Server Mode
**Terminal 1 (GPU Server):**
```bash
python server_gpu.py
```

**Terminal 2 (Client Camera):**
```bash
# Streams RTSP IP Camera to GPU Server (default source: rtsp://192.168.1.18:554/stream1)
python client_camera.py --server http://localhost:8000

# Or using local webcam
python client_camera.py --source 0 --server http://localhost:8000
```

---

## 📊 Excel Attendance Schema

Each daily workbook (`attendance/Attendance_YYYY-MM-DD.xlsx`) includes styled headers, auto-fitted columns, and clean formatting:

| S.No | Date | Name | In-Time | Out-Time | Duration | Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `1` | `2026-10-05` | Vishal | `09:15:22` | `18:10:45` | `08h 55m` | `Present` |
| `2` | `2026-10-05` | Abhinav | `09:22:04` | `-` | `-` | `Present` |

---

## 📜 License
MIT License. Built for automated attendance tracking and computer vision workflows.
