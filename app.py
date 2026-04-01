import time
from pathlib import Path
from threading import Lock, Thread

import av
import cv2
import imutils
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

try:
    import dlib
    from imutils import face_utils

    DLIB_AVAILABLE = True
except Exception:
    dlib = None
    face_utils = None
    DLIB_AVAILABLE = False

import light_remover as lr
import make_train_data as mtd
import ringing_alarm as alarm


st.set_page_config(page_title="Drowsiness Detection System", page_icon="😴", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
SHAPE_PREDICTOR_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
HAAR_FACE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
HAAR_EYE_PATH = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"

# Browser camera needs STUN for peer connection in most setups.
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


def eye_aspect_ratio(eye: np.ndarray) -> float:
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = Lock()

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.ear_thresh = 20.0
        self.ear_consec_frames = 20
        self.counter = 0
        self.timer_flag = False
        self.alarm_flag = False
        self.alarm_count = 0
        self.running_time = 0.0
        self.prev_term = 0.0
        self.start_closing = 0.0

        np.random.seed(9)
        self.power, self.nomal, self.short = mtd.start(25)

        self.last_ear = 0.0
        self.last_status = "No face detected"
        self.last_alarm_level = "-"
        self.last_alarm_time = 0.0

        self.frame_index = 0
        self.process_every = 3
        self.target_width = 480

    def _fire_alarm(self, result: int) -> None:
        now = time.time()
        if now - self.last_alarm_time < 1.5:
            return
        self.last_alarm_time = now
        Thread(target=alarm.select_alarm, args=(result,), daemon=True).start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = imutils.resize(image, width=self.target_width)

        _, gray = lr.light_removing(image)

        self.frame_index += 1
        if (self.frame_index % self.process_every) != 0:
            with self.lock:
                cv2.putText(
                    image,
                    f"EAR: {self.last_ear:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"Status: {self.last_status}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        rects = self.detector(gray, 0)
        status = "No face detected"
        ear_value = 0.0

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[self.l_start : self.l_end]
            right_eye = shape[self.r_start : self.r_end]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            both_ear = (left_ear + right_ear) * 500.0

            ear_value = both_ear

            left_hull = cv2.convexHull(left_eye)
            right_hull = cv2.convexHull(right_eye)
            cv2.drawContours(image, [left_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(image, [right_hull], -1, (0, 255, 0), 1)

            if both_ear < self.ear_thresh:
                if not self.timer_flag:
                    self.start_closing = time.perf_counter()
                    self.timer_flag = True
                self.counter += 1

                if self.counter >= self.ear_consec_frames:
                    closing_time = round(time.perf_counter() - self.start_closing, 3)

                    if closing_time >= self.running_time:
                        if self.running_time == 0:
                            cur_term = time.perf_counter()
                            opened_eyes_time = round((cur_term - self.prev_term), 3)
                            self.prev_term = cur_term
                            self.running_time = 1.75
                        else:
                            opened_eyes_time = 0.0

                        self.running_time += 2
                        self.alarm_flag = True
                        self.alarm_count += 1

                        result = mtd.run(
                            [opened_eyes_time, closing_time * 10],
                            self.power,
                            self.nomal,
                            self.short,
                        )
                        self.last_alarm_level = str(result)
                        self._fire_alarm(result)

                    status = "Drowsy"
                else:
                    status = "Blinking"
            else:
                self.counter = 0
                self.timer_flag = False
                self.running_time = 0
                self.alarm_flag = False
                status = "Awake"

            break

        with self.lock:
            self.last_ear = ear_value
            self.last_status = status

        cv2.putText(
            image,
            f"EAR: {ear_value:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"Status: {status}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Alarm Count: {self.alarm_count}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


class HaarDrowsinessProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = Lock()

        self.face_cascade = cv2.CascadeClassifier(str(HAAR_FACE_PATH))
        self.eye_cascade = cv2.CascadeClassifier(str(HAAR_EYE_PATH))

        self.closed_eye_frames = 0
        self.closed_eye_thresh = 18
        self.alarm_count = 0
        self.last_alarm_time = 0.0

        self.last_status = "No face detected"
        self.last_eye_count = 0
        self.target_width = 480

    def _fire_alarm(self) -> None:
        now = time.time()
        if now - self.last_alarm_time < 1.5:
            return
        self.last_alarm_time = now
        Thread(target=alarm.select_alarm, args=(2,), daemon=True).start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = imutils.resize(image, width=self.target_width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        status = "No face detected"
        eye_count = 0

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 255), 2)

            roi_gray = gray[y : y + h, x : x + w]
            roi_color = image[y : y + h, x : x + w]

            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(18, 18),
            )
            eye_count = len(eyes)

            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

            if eye_count >= 2:
                self.closed_eye_frames = 0
                status = "Awake"
            elif eye_count == 1:
                self.closed_eye_frames += 1
                status = "Blinking"
            else:
                self.closed_eye_frames += 1
                if self.closed_eye_frames >= self.closed_eye_thresh:
                    status = "Drowsy"
                    self.alarm_count += 1
                    self._fire_alarm()
                else:
                    status = "Eyes closed"

        with self.lock:
            self.last_status = status
            self.last_eye_count = eye_count

        cv2.putText(
            image,
            f"Status: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Detected eyes: {eye_count}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"Alarm Count: {self.alarm_count}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.title("Drowsiness Detection System 😴")
st.write("Live webcam detection runs directly inside Streamlit.")

if DLIB_AVAILABLE and not SHAPE_PREDICTOR_PATH.exists():
    st.warning(
        "dlib is available but shape_predictor_68_face_landmarks.dat is missing. "
        "Falling back to OpenCV Haar eye detection."
    )

if DLIB_AVAILABLE and SHAPE_PREDICTOR_PATH.exists():
    processor_factory = DrowsinessProcessor
    st.success("Mode: dlib landmark detector (high accuracy)")
else:
    processor_factory = HaarDrowsinessProcessor
    st.warning("Mode: OpenCV Haar fallback (works without dlib, lower accuracy)")

webrtc_ctx = webrtc_streamer(
    key="drowsiness-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=processor_factory,
    async_processing=True,
    desired_playing_state=True,
)

status_col, metric_col, info_col = st.columns(3)

if webrtc_ctx.state.playing:
    status_col.success("Camera is running")
else:
    status_col.info("Click START and choose your webcam from SELECT DEVICE")

processor = webrtc_ctx.video_processor
if processor:
    with processor.lock:
        if hasattr(processor, "last_ear"):
            metric_col.metric("EAR", f"{processor.last_ear:.2f}")
        else:
            metric_col.metric("Detected Eyes", str(processor.last_eye_count))
        info_col.metric("Alarm Count", str(processor.alarm_count))
        st.write(f"Status: {processor.last_status}")
        if hasattr(processor, "last_alarm_level"):
            st.write(f"Last alarm level: {processor.last_alarm_level}")

st.caption(
    "If the video panel stays blank, click START, pick a camera in SELECT DEVICE, "
    "allow browser webcam permission, then refresh once."
)
