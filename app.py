import streamlit as st
import cv2
import numpy as np
import os
import csv
import time
import math
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ---------- PAGE ----------
st.set_page_config(page_title="EyeStrain Data Collection", layout="centered")


# ---------- FOLDERS ----------
os.makedirs("metadata", exist_ok=True)

COUNTER_FILE = "participant_counter.txt"
MODEL_PATH = "face_landmarker.task"

MASTER_CSV = "metadata/participant_master.csv"
MORNING_CSV = "metadata/morning_session.csv"
EVENING_CSV = "metadata/evening_session.csv"


# ---------- CSV SETUP ----------
if not os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE, "w") as f:
        f.write("0")


def create_csv(file_name, header):
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            csv.writer(f).writerow(header)


create_csv(
    MASTER_CSV,
    ["Participant_ID", "Age", "Gender", "Spectacles", "Date_Time"]
)

header = [
    "Participant_ID", "Session", "Video_Name", "Duration",
    "Avg_Brightness", "Avg_Blur", "Avg_EAR",
    "Blink_Count", "Blink_Rate_Per_Minute",
    "Avg_Gaze_X", "Avg_Gaze_Y", "Date_Time"
]

create_csv(MORNING_CSV, header)
create_csv(EVENING_CSV, header)


# ---------- CONSENT ----------
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if not st.session_state.consent_given:
    st.title("Participant Consent Form")

    st.write("""
    This study collects ocular stress indicators such as blink dynamics,
    eye movement features, and digital eye strain-related information
    for PhD research purposes.

    Participation is voluntary. You may stop at any time.
    Camera access is used only for blink and eye movement feature extraction.
    """)

    consent = st.checkbox(
        "I have read the information above and voluntarily agree to participate."
    )

    if st.button("I Agree and Continue"):
        if consent:
            st.session_state.consent_given = True
            st.rerun()
        else:
            st.warning("Please provide consent to continue.")

    st.stop()


# ---------- FUNCTIONS ----------
def get_participant_id():
    with open(COUNTER_FILE, "r") as f:
        count = int(f.read().strip())

    count += 1

    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))

    return f"P{count:03d}"


def distance(p1, p2):
    return math.dist(p1, p2)


def calculate_ear(points):
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)


# ---------- WEBRTC PROCESSOR ----------
class EyeStrainProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmarker = create_landmarker()

        self.blink_count = 0
        self.eye_closed = False

        self.ear_values = []
        self.brightness_values = []
        self.blur_values = []
        self.gaze_x_values = []
        self.gaze_y_values = []

        self.EAR_THRESHOLD = 0.22

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w, _ = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

        self.brightness_values.append(brightness)
        self.blur_values.append(blur_value)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_points = []
            right_points = []

            for idx in self.LEFT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_points.append((x, y))
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            for idx in self.RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_points.append((x, y))
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            avg_ear = (left_ear + right_ear) / 2.0

            self.ear_values.append(avg_ear)

            if avg_ear < self.EAR_THRESHOLD:
                self.eye_closed = True
            else:
                if self.eye_closed:
                    self.blink_count += 1
                    self.eye_closed = False

            all_eye_points = left_points + right_points
            gaze_x = np.mean([p[0] for p in all_eye_points])
            gaze_y = np.mean([p[1] for p in all_eye_points])

            self.gaze_x_values.append(gaze_x)
            self.gaze_y_values.append(gaze_y)

            cv2.putText(
                img,
                f"Blinks: {self.blink_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                img,
                "Face not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def save_features(participant_id, session_label, duration_seconds, processor):
    csv_file = MORNING_CSV if session_label == "Baseline" else EVENING_CSV
    video_name = f"{participant_id}_{session_label}_browser_capture"

    avg_brightness = round(np.mean(processor.brightness_values), 2) if processor.brightness_values else 0
    avg_blur = round(np.mean(processor.blur_values), 2) if processor.blur_values else 0
    avg_ear = round(np.mean(processor.ear_values), 4) if processor.ear_values else 0
    avg_gaze_x = round(np.mean(processor.gaze_x_values), 2) if processor.gaze_x_values else 0
    avg_gaze_y = round(np.mean(processor.gaze_y_values), 2) if processor.gaze_y_values else 0

    blink_rate = round((processor.blink_count / duration_seconds) * 60, 2)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            participant_id,
            session_label,
            video_name,
            duration_seconds,
            avg_brightness,
            avg_blur,
            avg_ear,
            processor.blink_count,
            blink_rate,
            avg_gaze_x,
            avg_gaze_y,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

    return video_name


# ---------- MAIN APP ----------
st.title("EyeStrain Smart Video Data Collection System")

if not os.path.exists(MODEL_PATH):
    st.error("face_landmarker.task file not found. Upload it to GitHub beside app.py.")
    st.stop()

age = st.number_input("Age", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
spectacles = st.selectbox("Wearing spectacles?", ["Yes", "No"])

session = st.radio(
    "Select Session",
    ["Morning Session - Baseline", "Evening Session - Fatigue"]
)

duration = st.selectbox("Recording Duration", [30, 60, 120, 180, 300])

if "participant_id" not in st.session_state:
    st.session_state.participant_id = None

if st.button("Register Participant"):
    participant_id = get_participant_id()
    st.session_state.participant_id = participant_id

    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(MASTER_CSV, "a", newline="") as f:
        csv.writer(f).writerow([participant_id, age, gender, spectacles, date_time])

    st.success(f"Participant registered successfully. Participant ID: {participant_id}")


if st.session_state.participant_id:
    participant_id = st.session_state.participant_id
    label = "Baseline" if "Morning" in session else "Fatigue"

    st.subheader("Camera Recording")
    st.info("Click START below and allow camera permission in your browser.")

    ctx = webrtc_streamer(
        key=f"eyestrain-{participant_id}-{label}",
        video_processor_factory=EyeStrainProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.warning(f"Keep your face visible for approximately {duration} seconds.")

    if ctx.video_processor:
        if st.button("Save Recording Features"):
            video_name = save_features(
                participant_id,
                label,
                duration,
                ctx.video_processor
            )

            st.success("Features saved successfully.")
            st.write("Participant ID:", participant_id)
            st.write("Session:", label)
            st.write("Video Name:", video_name)
            st.write("Blink Count:", ctx.video_processor.blink_count)
else:
    st.info("Please register participant before starting camera recording.")
