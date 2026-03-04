import queue
import time
import urllib.request
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Blossom — Video Reaction Platform", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_COLORS = {
    "happy": "#FFD700",
    "sad": "#6495ED",
    "angry": "#FF4500",
    "surprise": "#FF69B4",
    "fear": "#9370DB",
    "disgust": "#3CB371",
    "neutral": "#A9A9A9",
}

EMOTION_EMOJI = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "😐",
}

# MediaPipe GestureRecognizer category names → display names
GESTURE_MAP = {
    "Thumb_Up": "Thumbs Up",
    "Thumb_Down": "Thumbs Down",
    "Open_Palm": "Open Palm",
    "Closed_Fist": "Fist",
    "Pointing_Up": "Pointing",
    "Victory": "Peace",
    "ILoveYou": "ILY",
    "None": "None",
}

GESTURE_EMOJI = {
    "Thumbs Up": "👍",
    "Thumbs Down": "👎",
    "Open Palm": "🖐",
    "Fist": "✊",
    "Pointing": "☝️",
    "Peace": "✌️",
    "ILY": "🤟",
    "None": "—",
}

GESTURE_SENTIMENT = {
    "Thumbs Up": "positive",
    "Thumbs Down": "negative",
    "Open Palm": "open / neutral",
    "Fist": "tense",
    "Pointing": "engaged",
    "Peace": "relaxed / positive",
    "ILY": "enthusiastic",
    "None": "—",
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

SAMPLE_VIDEO_URL = "https://www.w3schools.com/html/mov_bbb.mp4"
CAPTION = "Sample video: Big Buck Bunny (public domain) — testing video streaming functionality"

# ── Model paths ────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent / "models"
_FACE_MODEL = _MODEL_DIR / "face_landmarker.task"
_GESTURE_MODEL = _MODEL_DIR / "gesture_recognizer.task"

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)


@st.cache_resource(show_spinner="Downloading MediaPipe models (first run only)…")
def _ensure_models() -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(exist_ok=True)
    if not _FACE_MODEL.exists():
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL)
    if not _GESTURE_MODEL.exists():
        urllib.request.urlretrieve(_GESTURE_MODEL_URL, _GESTURE_MODEL)
    return _FACE_MODEL, _GESTURE_MODEL


# ── Session state ──────────────────────────────────────────────────────────────
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []


# ── Hand landmark drawing connections ─────────────────────────────────────────
_HAND_CONNECTIONS = [
    (c.start, c.end) for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
]

_FACE_CONTOUR_CONNECTIONS = [
    (c.start, c.end) for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
]


def _draw_hand(img: np.ndarray, hand_landmarks) -> None:
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in _HAND_CONNECTIONS:
        cv2.line(img, pts[start], pts[end], (0, 220, 90), 2)
    for pt in pts:
        cv2.circle(img, pt, 4, (255, 255, 255), -1)
        cv2.circle(img, pt, 4, (0, 180, 60), 1)


def _draw_face(img: np.ndarray, face_landmarks) -> None:
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
    for start, end in _FACE_CONTOUR_CONNECTIONS:
        cv2.line(img, pts[start], pts[end], (100, 200, 255), 1)


# ── Emotion from face blendshapes ──────────────────────────────────────────────
def _blendshapes_to_emotions(face_result) -> tuple[dict, str]:
    if not face_result.face_blendshapes:
        return {e: 0.0 for e in EMOTIONS}, "neutral"

    bs = {b.category_name: b.score for b in face_result.face_blendshapes[0]}

    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    brow_inner_up = bs.get("browInnerUp", 0)
    jaw_open = bs.get("jawOpen", 0)
    eye_wide = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    nose_sneer = (bs.get("noseSneerLeft", 0) + bs.get("noseSneerRight", 0)) / 2
    cheek_squint = (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0)) / 2

    happy = min(100.0, (smile * 0.65 + cheek_squint * 0.35) * 160)
    surprise = min(100.0, (jaw_open * 0.5 + eye_wide * 0.3 + brow_inner_up * 0.2) * 160)
    angry = min(100.0, (brow_down * 0.65 + frown * 0.35) * 160)
    sad = min(100.0, (frown * 0.55 + brow_inner_up * 0.25 + brow_down * 0.2) * 130)
    fear = min(100.0, (eye_wide * 0.45 + brow_inner_up * 0.35 + jaw_open * 0.2) * 130)
    disgust = min(100.0, (nose_sneer * 0.7 + frown * 0.3) * 160)
    neutral = max(0.0, 100.0 - (happy + surprise + angry + sad + fear + disgust) * 0.55)

    emotions = {
        "happy": happy, "surprise": surprise, "angry": angry,
        "sad": sad, "fear": fear, "disgust": disgust, "neutral": neutral,
    }
    return emotions, max(emotions, key=emotions.get)


# ── Combined emotion + gesture analysis ────────────────────────────────────────
def _combined_analysis(dominant_emotion: str, gesture: str) -> str:
    combos = {
        ("happy", "Thumbs Up"): "Enthusiastically positive — loving every moment.",
        ("happy", "Peace"): "Joyful and relaxed — clearly enjoying it.",
        ("happy", "Fist"): "Excited and pumped up!",
        ("happy", "ILY"): "Over the moon — an extremely enthusiastic reaction.",
        ("surprise", "Open Palm"): "Genuinely shocked — a strong visceral reaction.",
        ("surprise", "Thumbs Up"): "Pleasantly surprised — an unexpected delight.",
        ("sad", "Thumbs Down"): "Clear disappointment — not what was hoped for.",
        ("angry", "Fist"): "Frustrated — a strong negative response.",
        ("angry", "Thumbs Down"): "Strongly disapproves — deeply unsatisfied.",
        ("fear", "Open Palm"): "Startled or anxious — feeling unsettled.",
        ("neutral", "Thumbs Up"): "Mild approval — content but not excited.",
        ("neutral", "Thumbs Down"): "Passively dislikes — low engagement.",
        ("neutral", "Pointing"): "Curious and attentive — engaged but reserved.",
        ("disgust", "Thumbs Down"): "Visibly unimpressed — strong negative signal.",
        ("disgust", "Fist"): "Revulsion turning to tension.",
    }
    key = (dominant_emotion, gesture)
    if key in combos:
        return combos[key]
    sent = GESTURE_SENTIMENT.get(gesture, "neutral")
    if "positive" in sent or "enthusiastic" in sent:
        return f"Positive body language with {dominant_emotion} expression — encouraging signal."
    if "negative" in sent:
        return f"Negative gesture with {dominant_emotion} expression — possible mismatch worth noting."
    if sent == "tense":
        return f"Tense gesture alongside {dominant_emotion} — heightened emotional state."
    return f"{dominant_emotion.capitalize()} expression detected with {gesture.lower()} gesture."


# ── Video streaming ────────────────────────────────────────────────────────────
def stream_video(video_source, caption=""):
    st.video(video_source)
    if caption:
        st.caption(caption)


# ── Emotion + Gesture detector (runs in WebRTC thread) ─────────────────────────
class EmotionDetector(VideoProcessorBase):
    """
    Uses MediaPipe Tasks API:
      - FaceLandmarker with face blendshapes → emotion scores
      - GestureRecognizer → named hand gestures (Thumb_Up, Victory, etc.)
    """

    def __init__(self):
        self.result_queue: queue.Queue = queue.Queue()
        face_path, gesture_path = _ensure_models()
        self._last_ts_ms: int = 0

        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(face_path)),
            output_face_blendshapes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

        gesture_opts = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(gesture_path)),
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(gesture_opts)

    def __del__(self):
        self._face_landmarker.close()
        self._gesture_recognizer.close()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Timestamps must be strictly monotonically increasing
        ts_ms = max(int(time.time() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        )

        face_result = self._face_landmarker.detect_for_video(mp_img, ts_ms)
        gesture_result = self._gesture_recognizer.recognize_for_video(mp_img, ts_ms)

        emotions, dominant = _blendshapes_to_emotions(face_result)
        gesture = "None"

        # Draw face contours
        if face_result.face_landmarks:
            _draw_face(img, face_result.face_landmarks[0])

        # Draw hand skeleton + extract gesture
        if gesture_result.hand_landmarks:
            _draw_hand(img, gesture_result.hand_landmarks[0])
            if gesture_result.gestures:
                raw = gesture_result.gestures[0][0].category_name
                gesture = GESTURE_MAP.get(raw, raw)

        self.result_queue.put_nowait({
            "timestamp": time.time(),
            "emotions": emotions,
            "dominant": dominant,
            "gesture": gesture,
        })

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Emotion timeline chart ─────────────────────────────────────────────────────
def render_emotion_chart(history: list):
    if not history:
        st.info("Start the webcam above to begin tracking emotions in real time.")
        return

    t0 = history[0]["timestamp"]
    df = pd.DataFrame([
        {"time": round(h["timestamp"] - t0, 1), **h["emotions"]}
        for h in history
    ])

    fig = go.Figure()
    for emotion in EMOTIONS:
        if emotion in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[emotion],
                name=emotion.capitalize(),
                line=dict(color=EMOTION_COLORS[emotion], width=2),
                mode="lines",
            ))

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, use_container_width=True)


# ── Gesture timeline chart ─────────────────────────────────────────────────────
def render_gesture_timeline(history: list):
    gesture_frames = [h for h in history if h.get("gesture", "None") != "None"]
    if not gesture_frames:
        st.info("Show your hands to the webcam to see gesture tracking.")
        return

    t0 = history[0]["timestamp"]
    df = pd.DataFrame([
        {"time": round(h["timestamp"] - t0, 1), "gesture": h["gesture"]}
        for h in gesture_frames
    ])

    fig = go.Figure(go.Scatter(
        x=df["time"],
        y=df["gesture"],
        mode="markers+lines",
        marker=dict(size=8, color="#7B68EE"),
        line=dict(color="#7B68EE", width=1, dash="dot"),
        hovertemplate="%{y}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Gesture",
        margin=dict(l=20, r=20, t=20, b=20),
        height=220,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, use_container_width=True)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Blossom — Video Reaction Platform")
st.caption("Watch content on the left, react naturally — facial expressions and hand gestures are captured and analysed in real time.")
st.divider()

st_autorefresh(interval=2000, key="emotion_refresh")

# ── Row 1: video + webcam ──────────────────────────────────────────────────────
col_video, col_cam = st.columns(2, gap="large")

with col_video:
    st.subheader("Content")
    stream_video(SAMPLE_VIDEO_URL, CAPTION)

with col_cam:
    st.subheader("Reaction Capture")

    ctx = webrtc_streamer(
        key="emotion-detector",
        video_processor_factory=EmotionDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        while not ctx.video_processor.result_queue.empty():
            st.session_state.emotion_history.append(
                ctx.video_processor.result_queue.get_nowait()
            )

    if st.session_state.emotion_history:
        latest = st.session_state.emotion_history[-1]
        dominant = latest["dominant"]
        score = latest["emotions"][dominant]
        gesture = latest.get("gesture", "None")

        col_e, col_g = st.columns(2)
        with col_e:
            st.metric(
                label="Detected Emotion",
                value=f"{EMOTION_EMOJI.get(dominant, '')} {dominant.capitalize()}",
                delta=f"{score:.1f} score",
            )
        with col_g:
            g_label = f"{GESTURE_EMOJI.get(gesture, '')} {gesture}" if gesture != "None" else "— None"
            st.metric(
                label="Hand Gesture",
                value=g_label,
                delta=GESTURE_SENTIMENT.get(gesture, ""),
            )

        # Current emotion breakdown bar chart
        emotion_vals = latest["emotions"]
        mini_df = pd.DataFrame({
            "Emotion": [e.capitalize() for e in EMOTIONS],
            "Score": [emotion_vals.get(e, 0) for e in EMOTIONS],
            "Color": [EMOTION_COLORS[e] for e in EMOTIONS],
        }).sort_values("Score", ascending=True)

        fig_bar = go.Figure(go.Bar(
            x=mini_df["Score"],
            y=mini_df["Emotion"],
            orientation="h",
            marker_color=mini_df["Color"],
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0, 100], title="Score"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=220,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Combined analysis callout
        if gesture != "None":
            st.info(f"**Reaction Analysis:** {_combined_analysis(dominant, gesture)}")

st.divider()

# ── Row 2: timelines ───────────────────────────────────────────────────────────
col_tl, col_gl = st.columns([3, 2])

with col_tl:
    st.subheader("Emotion Timeline")
    render_emotion_chart(st.session_state.emotion_history)

with col_gl:
    st.subheader("Gesture Timeline")
    render_gesture_timeline(st.session_state.emotion_history)

if st.session_state.emotion_history:
    if st.button("Clear Session Data"):
        st.session_state.emotion_history = []
        st.rerun()
