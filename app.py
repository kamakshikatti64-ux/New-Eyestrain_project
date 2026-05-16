import streamlit as st
import cv2
import numpy as np
import os
import csv
import time
import math
from datetime import datetime

#import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision


# ---------- FOLDERS ----------
os.makedirs("videos/Morning", exist_ok=True)
os.makedirs("videos/Evening", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

COUNTER_FILE = "participant_counter.txt"
MODEL_PATH = "face_landmarker.task"

MASTER_CSV = "metadata/participant_master.csv"
MORNING_CSV = "metadata/morning_session.csv"
EVENING_CSV = "metadata/evening_session.csv"


if not os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE, "w") as f:
        f.write("0")


def create_csv(file_name, header):
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            csv.writer(f).writerow(header)


create_csv(MASTER_CSV, ["Participant_ID", "Age", "Gender", "Spectacles", "Date_Time"])

header = [
    "Participant_ID", "Session", "Video_Name", "Duration",
    "Avg_Brightness", "Avg_Blur", "Avg_EAR",
    "Blink_Count", "Blink_Rate_Per_Minute",
    "Avg_Gaze_X", "Avg_Gaze_Y", "Date_Time"
]

create_csv(MORNING_CSV, header)
create_csv(EVENING_CSV, header)


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
    return (vertical1 + vertical2) / (2.0 * horizontal)


def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)


def record_video(participant_id, session_label, duration_seconds):

    if not os.path.exists(MODEL_PATH):
        st.error("face_landmarker.task file not found in this folder.")
        return "Model file missing."

    if session_label == "Baseline":
        folder = "videos/Morning"
        csv_file = MORNING_CSV
        video_name = f"{participant_id}_Baseline.avi"
    else:
        folder = "videos/Evening"
        csv_file = EVENING_CSV
        video_name = f"{participant_id}_Fatigue.avi"

    video_path = os.path.join(folder, video_name)

    cap = cv2.VideoCapture(0)
    landmarker = create_landmarker()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None

    valid_time = 0
    last_time = time.time()
    recording_started = False

    brightness_values = []
    blur_values = []
    ear_values = []
    gaze_x_values = []
    gaze_y_values = []

    blink_count = 0
    eye_closed = False

    EAR_THRESHOLD = 0.22
    BLUR_THRESHOLD = 100

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    NOSE_TIP = 1

    while True:
        success, frame = cap.read()

        if not success:
            break

        h, w, _ = frame.shape
        center_x = w // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        quality_ok = True
        messages = []

        if brightness < 80:
            quality_ok = False
            messages.append("Increase room lighting")
        elif brightness > 200:
            quality_ok = False
            messages.append("Reduce excess light")

        if blur_value < BLUR_THRESHOLD:
            quality_ok = False
            messages.append("Camera is blurry")

        if not result.face_landmarks:
            quality_ok = False
            messages.append("Face not detected")
        else:
            landmarks = result.face_landmarks[0]

            nose_x = int(landmarks[NOSE_TIP].x * w)
            difference = nose_x - center_x

            if difference < -70:
                quality_ok = False
                messages.append("Move slightly right")
            elif difference > 70:
                quality_ok = False
                messages.append("Move slightly left")

            left_points = []
            right_points = []

            for idx in LEFT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)

            if avg_ear < EAR_THRESHOLD:
                eye_closed = True
            else:
                if eye_closed:
                    blink_count += 1
                    eye_closed = False

            left_gaze_x = np.mean([p[0] for p in left_points])
            right_gaze_x = np.mean([p[0] for p in right_points])
            left_gaze_y = np.mean([p[1] for p in left_points])
            right_gaze_y = np.mean([p[1] for p in right_points])

            gaze_x_values.append((left_gaze_x + right_gaze_x) / 2)
            gaze_y_values.append((left_gaze_y + right_gaze_y) / 2)

        current_time = time.time()
        gap = current_time - last_time
        last_time = current_time

        if quality_ok:
            if not recording_started:
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                recording_started = True

            out.write(frame)
            valid_time += gap
            brightness_values.append(brightness)
            blur_values.append(blur_value)

            status = "Recording - Stay still"
            color = (0, 255, 0)
        else:
            status = "Paused - Adjust position"
            color = (0, 0, 255)

        remaining = duration_seconds - int(valid_time)

        cv2.putText(frame, f"{session_label} Session", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, status, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Valid Time: {int(valid_time)} sec", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Remaining: {remaining} sec", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y_pos = 220
        for msg in messages:
            cv2.putText(frame, msg, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 35

        cv2.imshow("EyeStrain Smart Recording System", frame)

        if valid_time >= duration_seconds:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if out is not None:
        out.release()

    cv2.destroyAllWindows()

    avg_brightness = round(np.mean(brightness_values), 2) if brightness_values else 0
    avg_blur = round(np.mean(blur_values), 2) if blur_values else 0
    avg_ear = round(np.mean(ear_values), 4) if ear_values else 0
    avg_gaze_x = round(np.mean(gaze_x_values), 2) if gaze_x_values else 0
    avg_gaze_y = round(np.mean(gaze_y_values), 2) if gaze_y_values else 0
    blink_rate = round((blink_count / duration_seconds) * 60, 2)

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
            blink_count,
            blink_rate,
            avg_gaze_x,
            avg_gaze_y,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

    return video_name


st.title("EyeStrain Smart Video Data Collection System")

age = st.number_input("Age", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
spectacles = st.selectbox("Wearing spectacles?", ["Yes", "No"])

session = st.radio(
    "Select Session",
    ["Morning Session - Baseline", "Evening Session - Fatigue"]
)

duration = st.selectbox("Recording Duration", [30, 60, 120, 180, 300])

if st.button("Start Recording"):
    participant_id = get_participant_id()
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(MASTER_CSV, "a", newline="") as f:
        csv.writer(f).writerow([participant_id, age, gender, spectacles, date_time])

    label = "Baseline" if "Morning" in session else "Fatigue"

    st.warning("Webcam window will open. Press q only to stop manually.")

    video_name = record_video(participant_id, label, duration)

    st.success("Recording completed and saved.")
    st.write("Participant ID:", participant_id)
    st.write("Session:", label)
    st.write("Video Name:", video_name)
