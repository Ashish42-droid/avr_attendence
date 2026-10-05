import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise Exception("Could not open webcam")

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
