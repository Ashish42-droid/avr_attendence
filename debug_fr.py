import sys
try:
    import face_recognition_models
    print("face_recognition_models import OK:", face_recognition_models.__file__)
except Exception as e:
    print("face_recognition_models error:", e)

try:
    import dlib
    print("dlib import OK:", dlib.__version__)
except Exception as e:
    print("dlib error:", e)

try:
    import face_recognition
    print("face_recognition import OK")
except Exception as e:
    print("face_recognition error:", e)
