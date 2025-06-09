import os
import cv2
import time
import numpy as np
import dlib
from tensorflow.keras.models import load_model

shape_x, shape_y = 96, 96
emotion_weights = {"surprise": 5, "happy": 4, "sad": 3, "angry": 2, "neutral": 1}
mapped_emotions = ["angry", "happy", "neutral", "sad", "surprise"]

model = load_model("emotion_tl2_model.h5", compile=False)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

prev_pupil = None

def emotion_weight(emotion):
    return emotion_weights.get(emotion, 0)

def detect_face(frame):
    if frame is None:
        return None, []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
    return gray, faces

def extract_face_rgb(frame, faces):
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (shape_x, shape_y)).astype(np.float32) / 255.0
        return np.reshape(resized, (1, shape_x, shape_y, 3))
    return None

def get_eye_center(eye_points, gray):
    x, y = zip(*eye_points)
    roi = gray[min(y):max(y), min(x):max(x)]
    if roi.size == 0:
        return (min(x), min(y))
    _, thresh = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        M = cv2.moments(max(contours, key=cv2.contourArea))
        if M['m00'] != 0:
            return (int(M['m10']/M['m00']) + min(x), int(M['m01']/M['m00']) + min(y))
    return (min(x)+(max(x)-min(x))//2, min(y)+(max(y)-min(y))//2)

def classify_movement(prev, curr):
    if not prev or not curr:
        return "UNKNOWN"
    dist = np.linalg.norm(np.array(curr) - np.array(prev))
    return "HIGH_FOCUS" if dist < 8 else "MEDIUM_FOCUS" if dist < 20 else "LOW_FOCUS"

def calculate_attention_score(movement, emotion_score):
    eye_score = 10 if movement == "HIGH_FOCUS" else 5 if movement == "MEDIUM_FOCUS" else 0
    return eye_score * 0.6 + emotion_score * 0.4

def analyze_frame_np(frame):
    if frame is None:
        return {"emotion": "Error", "attention": 0}

    try:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
        if len(faces) == 0:
            return {"emotion": "Unknown", "attention": 0}

        face_img = extract_face_rgb(frame, faces)
        if face_img is None:
            return {"emotion": "Unknown", "attention": 0}

        pred = model.predict(face_img, verbose=0)[0]
        idx_em = np.argmax(pred)
        emotion = mapped_emotions[idx_em]
        conf = pred[idx_em]

        conf_score = 2 if conf > 0.7 else 1 if conf > 0.5 else 0
        emotion_score = min(10, conf_score + emotion_weight(emotion))

        return {"emotion": emotion, "attention": emotion_score}

    except Exception:
        return {"emotion": "Error", "attention": 0}
