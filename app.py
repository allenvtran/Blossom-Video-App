import queue
import time

import av
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from deepface import DeepFace
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


# ── Video streaming ────────────────────────────────────────────────────────────
def stream_video(video_source, caption=""):
    """Display a video with an optional caption."""
    st.video(video_source)
    if caption:
        st.caption(caption)


# ── Emotion detection (runs in WebRTC thread) ──────────────────────────────────
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.result_queue: queue.Queue = queue.Queue()
        self._frame_count = 0
        self._analyze_every = 15  # run DeepFace every N frames to stay performant

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        if self._frame_count % self._analyze_every == 0:
            try:
                results = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if results:
                    face = results[0]
                    self.result_queue.put({
                        "timestamp": time.time(),
                        "emotions": face["emotion"],
                        "dominant": face["dominant_emotion"],
                    })
            except Exception:
                pass

        return frame


# ── Emotion timeline chart ─────────────────────────────────────────────────────
def render_emotion_chart(history: list):
    if not history:
        st.info("Start the webcam above to begin tracking emotions in real time.")
        return

    t0 = history[0]["timestamp"]
    rows = [
        {"time": round(h["timestamp"] - t0, 1), **h["emotions"]}
        for h in history
    ]
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for emotion in EMOTIONS:
        if emotion in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df[emotion],
                name=emotion.capitalize(),
                line=dict(color=EMOTION_COLORS[emotion], width=2),
                mode="lines",
                fill="none",
            ))

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence (%)",
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
st.caption("Watch content on the left, react naturally — your emotions are captured and analysed in real time.")

st.divider()

# Auto-refresh every 2 s so the chart and metrics update without user interaction
st_autorefresh(interval=2000, key="emotion_refresh")

# ── Row 1: video player + webcam side by side ──────────────────────────────────
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

    # Drain results produced in the WebRTC thread into session state
    if ctx.video_processor:
        while not ctx.video_processor.result_queue.empty():
            st.session_state.emotion_history.append(
                ctx.video_processor.result_queue.get_nowait()
            )

    # Live dominant emotion metric
    if st.session_state.emotion_history:
        latest = st.session_state.emotion_history[-1]
        dominant = latest["dominant"]
        confidence = latest["emotions"][dominant]

        st.metric(
            label="Detected Emotion",
            value=f"{EMOTION_EMOJI.get(dominant, '')} {dominant.capitalize()}",
            delta=f"{confidence:.1f}% confidence",
        )

        # Mini bar chart of current frame emotion breakdown
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
            xaxis=dict(range=[0, 100], title="Confidence (%)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=220,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Row 2: Emotion timeline ────────────────────────────────────────────────────
st.subheader("Emotion Timeline")
render_emotion_chart(st.session_state.emotion_history)

# Clear session button
if st.session_state.emotion_history:
    if st.button("Clear Session Data"):
        st.session_state.emotion_history = []
        st.rerun()
