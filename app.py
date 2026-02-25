import queue
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

SAMPLE_VIDEO_URL = "https://www.w3schools.com/html/mov_bbb.mp4"
# replace the video url with the video url you want to stream
CAPTION = "Sample video: Big Buck Bunny (public domain) — testing video streaming functionality"

# ── Session state ──────────────────────────────────────────────────────────────
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# ── MediaPipe setup (module-level, shared across frames) ───────────────────────
_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


# ── Video streaming ────────────────────────────────────────────────────────────
def stream_video(video_source, caption=""):
    """Display a video with an optional caption."""
    st.video(video_source)
    if caption:
        st.caption(caption)


# ── Emotion detector (runs in WebRTC thread) ───────────────────────────────────
class EmotionDetector(VideoProcessorBase):
    """
    Uses MediaPipe FaceMesh landmarks to derive emotion scores geometrically:
      - Mouth aspect ratio  → surprise / happy
      - Lip corner height   → happy / sad
      - Eyebrow raise       → surprise / fear
    No TensorFlow required.
    """

    def __init__(self):
        self.result_queue: queue.Queue = queue.Queue()
        self._face_mesh = _mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _score_emotions(self, landmarks, w: int, h: int) -> tuple[dict, str]:
        def pt(idx) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        # Mouth openness: vertical gap vs. width
        mouth_open = np.linalg.norm(pt(13) - pt(14))
        mouth_width = np.linalg.norm(pt(61) - pt(291)) + 1e-6
        mar = mouth_open / mouth_width  # mouth aspect ratio

        # Smile: mid-lip y vs. corner y (positive = corners higher = smile)
        mouth_mid_y = (pt(13)[1] + pt(14)[1]) / 2
        corners_y = (pt(61)[1] + pt(291)[1]) / 2
        smile = (mouth_mid_y - corners_y) / (h + 1e-6)

        # Eyebrow raise: brow landmark above eye landmark
        left_raise = (pt(159)[1] - pt(66)[1]) / (h + 1e-6)
        right_raise = (pt(386)[1] - pt(296)[1]) / (h + 1e-6)
        brow_raise = (left_raise + right_raise) / 2

        happy = float(np.clip(smile * 2000 + 45, 0, 100))
        surprise = float(np.clip(mar * 150 + brow_raise * 3000, 0, 100))
        sad = float(np.clip(-smile * 1500 + 20, 0, 100))
        fear = float(np.clip(brow_raise * 2000 - 10, 0, 100))
        angry = float(np.clip(-brow_raise * 1500 + 15, 0, 100))
        disgust = float(np.clip(-smile * 800 + brow_raise * 500, 0, 100))
        neutral = float(np.clip(100 - happy * 0.6 - surprise * 0.3 - sad * 0.2, 0, 100))

        emotions = {
            "happy": happy,
            "sad": sad,
            "angry": angry,
            "surprise": surprise,
            "fear": fear,
            "disgust": disgust,
            "neutral": neutral,
        }
        return emotions, max(emotions, key=emotions.get)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        results = self._face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0]
            emotions, dominant = self._score_emotions(face_lms.landmark, w, h)
            self.result_queue.put_nowait({
                "timestamp": time.time(),
                "emotions": emotions,
                "dominant": dominant,
            })
            _mp_drawing.draw_landmarks(
                img,
                face_lms,
                _mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

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
                x=df["time"],
                y=df[emotion],
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
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, use_container_width=True)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Blossom — Video Reaction Platform")
st.caption("Watch content on the left, react naturally — your facial expressions are captured and analysed in real time.")
st.divider()

# Refresh every 2 s so the chart updates without user interaction
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

            st.metric(
                label="Detected Emotion",
                value=f"{EMOTION_EMOJI.get(dominant, '')} {dominant.capitalize()}",
                delta=f"{score:.1f} score",
            )

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

# ── Row 2: timeline ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Emotion Timeline")
render_emotion_chart(st.session_state.emotion_history)

if st.session_state.emotion_history:
    if st.button("Clear Session Data"):
        st.session_state.emotion_history = []
        st.rerun()
