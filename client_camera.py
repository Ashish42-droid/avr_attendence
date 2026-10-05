import cv2
import requests
import time
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Live Camera Face Recognition Client (DGX GPU Accelerated)")
    parser.add_argument("--source", type=str, default="rtsp://192.168.1.18:554/stream1", help="Camera source (0 for local webcam, or RTSP URL e.g. rtsp://192.168.1.18:554/stream1)")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="DGX Server URL (default: http://localhost:8000)")
    args = parser.parse_args()

    # Convert numeric camera index if applicable
    source = int(args.source) if args.source.isdigit() else args.source
    server_url = args.server.rstrip("/")

    print("=" * 60)
    print("LIVE CAMERA GPU FACE RECOGNITION CLIENT")
    print("=" * 60)
    print(f"Connecting to DGX GPU Server: {server_url}")

    # Health check
    try:
        res = requests.get(f"{server_url}/health", timeout=3)
        if res.status_code == 200:
            health = res.json()
            print(f"[+] DGX Server Online! GPU Active: {health.get('gpu')}, Device: {health.get('gpu_name')}, Faces Loaded: {health.get('faces_loaded')}")
        else:
            print(f"[!] Server returned status {res.status_code}")
    except Exception as e:
        print(f"[!] Could not connect to {server_url}/health: {e}")
        print("[*] Make sure the server is running on DGX and SSH port forwarding is active.")
        print("[*] Command: ssh -L 8000:localhost:8000 dgx-s-csjmu-ece-vishal@172.20.8.10")

    print(f"\nOpening Video Source: {source}...")
    if isinstance(source, str) and source.startswith("rtsp://"):
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        import numpy as np
        sources_to_try = [1, 0, 2] if source == 0 else [source, 1, 0, 2]
        seen = set()
        unique_sources = [x for x in sources_to_try if not (x in seen or seen.add(x))]
        cap = None

        print("[*] Probing for active camera...")
        for idx in unique_sources:
            for b_name, b_flag in [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]:
                c = cv2.VideoCapture(idx, b_flag)
                if c.isOpened():
                    for _ in range(5):
                        ret, test_f = c.read()
                        if ret and test_f is not None and np.mean(test_f) > 5:
                            print(f"[+] Connected to Camera Index {idx} ({b_name})!")
                            cap = c
                            break
                    if cap is not None:
                        break
                    c.release()
            if cap is not None:
                break

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {source}")
        print("[*] Tip: Make sure your webcam is plugged in and no other app (Zoom, Teams, etc.) is using it.")
        sys.exit(1)

    print("[+] Camera started! Press 'q' in the video window to exit.\n")

    frame_count = 0
    cached_faces = []
    last_inference_time = 0
    fps_time = time.time()
    fps = 0

    # Live Attendance Banner State
    banner_msg = None
    banner_color = (0, 255, 0)
    banner_expiry = 0
    consecutive_drops = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_drops += 1
            time.sleep(0.05)
            if consecutive_drops % 10 == 0:
                print(f"[!] Frame drop from camera ({consecutive_drops} drops). Retrying...")
            if consecutive_drops >= 30 and isinstance(source, str) and source.startswith("rtsp://"):
                print("[!] Stream interrupted. Attempting RTSP reconnection...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_drops = 0
                if not cap.isOpened():
                    print("[ERROR] Failed to reconnect to RTSP stream.")
            continue
        consecutive_drops = 0

        frame_count += 1

        # Calculate display FPS
        if frame_count % 10 == 0:
            fps = round(10 / (time.time() - fps_time), 1)
            fps_time = time.time()

        # Send frame to DGX GPU every 2nd frame for maximum responsiveness
        if frame_count % 2 == 0:
            try:
                # Downsample frame slightly for ultra-fast network transmission
                send_frame = cv2.resize(frame, (640, 480))
                _, encoded_img = cv2.imencode('.jpg', send_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Scale factors to map box back to original frame
                scale_y = frame.shape[0] / 480.0
                scale_x = frame.shape[1] / 640.0

                resp = requests.post(
                    f"{server_url}/recognize",
                    files={"file": ("frame.jpg", encoded_img.tobytes(), "image/jpeg")},
                    timeout=0.5
                )

                if resp.status_code == 200:
                    data = resp.json()
                    last_inference_time = data.get("inference_ms", 0)
                    raw_faces = data.get("faces", [])
                    
                    # Rescale bounding boxes
                    cached_faces = []
                    for f in raw_faces:
                        top, right, bottom, left = f["box"]
                        scaled_box = (
                            int(top * scale_y),
                            int(right * scale_x),
                            int(bottom * scale_y),
                            int(left * scale_x)
                        )
                        cached_faces.append({
                            "name": f["name"],
                            "box": scaled_box,
                            "distance": f["distance"],
                            "attendance": f.get("attendance")
                        })
                        if f["name"] != "Unknown":
                            print(f"[{time.strftime('%H:%M:%S')}] DETECTED: {f['name']} (dist: {f['distance']}) - GPU Inference: {last_inference_time}ms")

                        # Trigger on-screen banner if attendance was marked or updated
                        att = f.get("attendance")
                        if att and att.get("status") in ["MARKED_IN", "MARKED_OUT", "ALREADY_PRESENT"]:
                            banner_msg = att.get("message")
                            banner_color = tuple(att.get("color", [0, 255, 0]))
                            banner_expiry = time.time() + 3.5

            except Exception:
                pass  # Keep smooth live feed if individual frame request drops

        # Draw bounding boxes & labels on the live frame
        for f in cached_faces:
            top, right, bottom, left = f["box"]
            name = f["name"]
            is_match = (name != "Unknown")
            color = (0, 255, 0) if is_match else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label box
            label = f"{name}"
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Attendance Confirmation Banner
        if banner_msg and time.time() < banner_expiry:
            h, w = frame.shape[:2]
            banner_h = 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.line(frame, (0, banner_h - 2), (w, banner_h - 2), banner_color, 3)
            cv2.putText(
                frame,
                banner_msg,
                (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Overlay FPS & GPU Inference stats
        stats_text = f"FPS: {fps} | DGX GPU: {last_inference_time}ms"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Show live stream
        cv2.imshow("Live Face Recognition (Powered by DGX GPU)", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Client closed.")

if __name__ == "__main__":
    main()
