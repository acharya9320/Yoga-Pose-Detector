import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from PIL import Image
import tempfile
import base64
import os
import time
import plotly.graph_objects as go

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Yoga Pose Detector", layout="wide")

# ----------------------------------------------------
# UTILITY: Base64 Image Loader
# ----------------------------------------------------
def get_base64_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ----------------------------------------------------
# SESSION STATE for camera control
# ----------------------------------------------------
if "cam_running" not in st.session_state:
    st.session_state.cam_running = False
if "stop_request" not in st.session_state:
    st.session_state.stop_request = False


# ----------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.header("Controls")
flip_cam = st.sidebar.checkbox("Flip Camera", False)
show_land = st.sidebar.checkbox("Show Landmarks", True)

frame_skip = st.sidebar.slider(
    "Process every Nth frame", 1, 5, 2,
    help="हर N-वा फ्रेम प्रोसेस होगा — बड़ा N CPU बचाता है पर accuracy थोड़ी घटती है"
)

frame_limit = st.sidebar.slider(
    "Frame Limit", 50, 600, 150,
    help="कैमरा कुल कितने फ्रेम तक चलेगा"
)

st.sidebar.markdown("---")
theme_choice = st.sidebar.radio("Choose Theme", ["Day Mode", "Night Mode"])


# ----------------------------------------------------
# THEME + FIXES (Navbar + File-upload + Buttons)
# ----------------------------------------------------
if theme_choice == "Day Mode":
    bg = "#ffffff"
    text = "#0f172a"
    nav = "#03583D"
    card = "#f2ff00"
    plotly_template = "plotly_white"
    button_bg = "#edf2f7"
    button_color = "#0f172a"
    upload_bg = "#f7f7f7"
    upload_text = "#000000"
else:
    bg = "#0b1220"
    text = "#e2e8f0"
    nav = "#064e3b"
    card = "#111827"
    plotly_template = "plotly_dark"
    button_bg = "#1e293b"
    button_color = "#e2e8f0"
    upload_bg = "#1e293b"
    upload_text = "white"

# CSS FIX • File-uploader Dark Mode • Navbar Scroll • Button Colors
css_code = f"""
<style>

html {{ scroll-behavior: smooth; }}

[data-testid="stAppViewContainer"] {{
    background-color: {bg} !important;
}}

h1,h2,h3,h4,h5,p,div,label,span {{
    color: {text} !important;
}}

[data-testid="stSidebar"] {{
    background-color: {card} !important;
    color: {text} !important;
}}

.navbar {{
    background:{nav};
    padding:12px;
    text-align:center;
    border-radius:8px;
    margin-bottom:18px;
}}
.navbar a {{
    color:white !important;
    padding:10px 18px;
    text-decoration:none;
    font-weight:600;
}}
.navbar a:hover {{
    background:#2563EB;
    border-radius:6px;
}}

.stButton>button {{
    background-color: {button_bg} !important;
    color: {button_color} !important;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight:600;
}}

input[type="file"] {{
    background:{upload_bg} !important;
    color:{upload_text} !important;
    padding:10px;
    border-radius:8px;
}}

.footer {{
    padding:18px;
    text-align:center;
    color:{text};
    margin-top:30px;
}}
</style>
"""

st.markdown(css_code, unsafe_allow_html=True)


# ----------------------------------------------------
# NAVBAR + HERO
# ----------------------------------------------------
hero_img = "pic.png" if os.path.exists("pic.png") else "yoga.png"
bg64 = get_base64_image(hero_img)

st.markdown(
    f"""
    <div class="navbar">
      <a href="#home_section">Home</a>
      <a href="#video_section">Video Upload</a>
      <a href="#image_section">Image Detection</a>
      <a href="#about_section">About</a>
    </div>

    <div id="home_section" style="
        height:100vh;
        border-radius:12px;
        background-size:cover;
        background-position:center;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        background-image:url('data:image/png;base64,{bg64}');
    ">
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------
# LOAD MODEL (with safety)
# ----------------------------------------------------
@st.cache_resource
def load_pose_model():
    model = load_model("model.h5")

    labels = np.load("labels.npy")
    labels = [str(x) for x in labels]
    return model, labels


model, labels = load_pose_model()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


# ----------------------------------------------------
# FIX: NORMALIZE PROBABILITY (NO MORE >100% ACCURACY)
# ----------------------------------------------------
def normalize_prob(prob):
    s = np.sum(prob)
    if s == 0:
        return prob
    return prob / s  # ensures probabilities always sum to 1

# ----------------------------------------------------
# PIE CHART (used in all 3 sections)
# ----------------------------------------------------
def render_pie(placeholder, pose_name, conf_percent, title="Confidence"):
    conf_percent = float(max(0, min(conf_percent, 100)))  # never above 100
    other = 100 - conf_percent

    fig = go.Figure(
        data=[go.Pie(
            labels=[pose_name, "Others"],
            values=[conf_percent, other],
            hole=0.45,
            marker=dict(colors=["#16a34a", "#A80E0E"])
        )]
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="inside"
    )

    fig.update_layout(
        title=title,
        template=plotly_template,
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    placeholder.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# BAR — TOP-5 Predictions (optional + small)
# ----------------------------------------------------
def render_bar_top5(placeholder, prob, labels):
    prob = normalize_prob(prob)
    idxs = np.argsort(prob)[::-1][:5]
    top_labels = [labels[i] for i in idxs]
    top_vals = [prob[i] * 100 for i in idxs]

    fig = go.Figure(go.Bar(
        x=top_vals, y=top_labels,
        orientation="h",
        marker=dict(color="#16a34a"),
        text=[f"{v:.1f}%" for v in top_vals],
        textposition="outside"
    ))

    fig.update_layout(
        title="Top Predictions",
        template=plotly_template,
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis=dict(autorange="reversed")
    )

    placeholder.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# FINAL RESULT CARD (common)
# ----------------------------------------------------
def show_final_card(placeholder, heading, pose_name, conf):
    conf = float(max(0, min(conf, 100)))
    color = "#16a34a" if conf >= 75 else "#f59e0b" if conf >= 40 else "#ef4444"

    html = f"""
    <div style="background:{card}; padding:12px; border-radius:10px; margin-bottom:10px;">
        <div style="font-size:18px; font-weight:700; color:{color};">{heading}</div>
        <div style="margin-top:6px;"><b>Pose:</b> {pose_name}</div>
        <div style="margin-top:4px;"><b>Confidence:</b> {conf:.2f}%</div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------
# LIVE CAMERA SECTION
# ----------------------------------------------------
st.markdown("<h2 id='live_section'>Live Camera</h2>", unsafe_allow_html=True)

col_start, col_end = st.columns(2)
start_cam = col_start.button("Start Camera")
stop_cam = col_end.button("Stop Camera")

if start_cam:
    st.session_state.cam_running = True
if stop_cam:
    st.session_state.cam_running = False
    st.session_state.stop_request = True

cam_left, cam_right = st.columns([2, 1])

frame_display = cam_left.image(np.zeros((480, 640, 3), dtype=np.uint8))
live_result_box = cam_right.container()
live_pie_box = cam_right.container()

if st.session_state.cam_running:
    cap = cv2.VideoCapture(0)
    last_pose = "None"
    last_conf = 0.0
    avg_list = []
    smooth_buffer = []

    frame_count = 0

    while cap.isOpened() and not st.session_state.stop_request and frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        if flip_cam:
            frame = cv2.flip(frame, 1)

        if frame_count % frame_skip == 0:
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                pts = []
                for lm in result.pose_landmarks.landmark:
                    pts.append(lm.x - result.pose_landmarks.landmark[0].x)
                    pts.append(lm.y - result.pose_landmarks.landmark[0].y)

                pts = np.array(pts).reshape(1, -1)

                prob = model.predict(pts)[0]
                prob = normalize_prob(prob)      # <- FIX: never above 100%
                smooth_buffer.append(prob)

                if len(smooth_buffer) > 5:
                    smooth_buffer.pop(0)

                avg_prob = np.mean(smooth_buffer, axis=0)

                idx = np.argmax(avg_prob)
                last_pose = labels[idx]
                last_conf = avg_prob[idx] * 100
                avg_list.append(last_conf)

                if show_land:
                    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"{last_pose} ({last_conf:.1f}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 240, 0), 2)

        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        time.sleep(0.03)

    cap.release()
    st.session_state.cam_running = False

    avg_conf = np.mean(avg_list) if avg_list else last_conf

    show_final_card(live_result_box, "Live Camera Summary", last_pose, avg_conf)
    render_pie(live_pie_box, last_pose, avg_conf, title="Live Camera Confidence")


# ----------------------------------------------------
# VIDEO UPLOAD SECTION
# ----------------------------------------------------
st.markdown("<h2 id='video_section'>Video Upload</h2>", unsafe_allow_html=True)

video_file = st.file_uploader("Upload a video", type=["mp4", "gif"])

if video_file:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video_file.read())
    cap = cv2.VideoCapture(temp.name)

    vid_left, vid_right = st.columns([2, 1])
    vid_display = vid_left.image(np.zeros((480, 640, 3), dtype=np.uint8))
    vid_result_box = vid_right.container()
    vid_pie_box = vid_right.container()

    smooth = []
    avg_conf_list = []
    final_pose = "None"
    final_conf = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            pts = []
            for lm in result.pose_landmarks.landmark:
                pts.append(lm.x - result.pose_landmarks.landmark[0].x)
                pts.append(lm.y - result.pose_landmarks.landmark[0].y)

            pts = np.array(pts).reshape(1, -1)
            prob = model.predict(pts)[0]
            prob = normalize_prob(prob)

            smooth.append(prob)
            if len(smooth) > 5:
                smooth.pop(0)

            avg_prob = np.mean(smooth, axis=0)

            idx = np.argmax(avg_prob)
            final_pose = labels[idx]
            final_conf = avg_prob[idx] * 100
            avg_conf_list.append(final_conf)

            if show_land:
                mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        vid_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    avg_video_conf = np.mean(avg_conf_list) if avg_conf_list else final_conf

    show_final_card(vid_result_box, "Video Summary", final_pose, avg_video_conf)
    render_pie(vid_pie_box, final_pose, avg_video_conf, title="Video Confidence")
# ----------------------------------------------------
# IMAGE UPLOAD SECTION
# ----------------------------------------------------
st.markdown("<h2 id='image_section'>Image Upload</h2>", unsafe_allow_html=True)

# Fix: Image-uploader style for Night Mode
st.markdown(
    f"""
    <style>
    [data-testid="stFileUploadDropzone"] {{
        background-color: {card} !important;
        border: 2px dashed #4a5568 !important;
    }}
    [data-testid="stFileUploadDropzone"]::after {{
        color: {text} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

img_file = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    frame = np.array(img)
    frame = cv2.resize(frame, (640, 480))

    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_left, img_right = st.columns([2, 1])
    img_left.image(frame, caption="Uploaded Image", use_container_width=True)

    img_result_box = img_right.container()
    img_pie_box = img_right.container()

    if result.pose_landmarks:
        pts = []
        for lm in result.pose_landmarks.landmark:
            pts.append(lm.x - result.pose_landmarks.landmark[0].x)
            pts.append(lm.y - result.pose_landmarks.landmark[0].y)

        pts = np.array(pts).reshape(1, -1)

        prob = model.predict(pts)[0]
        prob = normalize_prob(prob)    # FIX: never above 100%

        idx = int(np.argmax(prob))
        pose_name = labels[idx]
        conf_val = prob[idx] * 100

        # Draw landmarks
        if show_land:
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        img_left.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Summary Card + PIE + BAR
        show_final_card(img_result_box, "Image Summary", pose_name, conf_val)
        render_pie(img_pie_box, pose_name, conf_val, title="Image Confidence")
        render_bar_top5(img_pie_box, prob, labels)

    else:
        img_result_box.warning("⚠ No pose detected — ensure full body is visible & lighting is proper.")


# ----------------------------------------------------
# ABOUT + FOOTER
# ----------------------------------------------------
st.markdown("<h2 id='about_section'>About</h2>", unsafe_allow_html=True)

st.write("""
यह एप्लिकेशन विशेष रूप से योग अभ्यास करने वालों के लिए तैयार किया गया है।  
इसका उद्देश्य है — आपको यह बताना कि आप किसी योगासन को कितनी सटीकता और आत्मविश्वास के साथ कर पा रहे हैं।

इस ऐप का उपयोग करते समय आप तीन तरीकों से अपना आसन जाँच सकते हैं —  
1) **Live Camera**  
2) **Video Upload**  
3) **Image Upload**  

एप्लिकेशन आपके पूरे शरीर के पोज़ को पढ़कर उसकी तुलना प्रशिक्षित योग-आसनों से करता है और बताता है कि —  
- आपने कौन-सा आसन किया,  
- वह आसन कितनी सटीकता (Confidence Score) से पहचाना गया,  
- और आपकी पोज़ पर आधारित एक संक्षिप्त विश्लेषण प्राप्त होता है।  

इस तरह, आप अपनी मुद्रा में सुधार कर सकते हैं, अभ्यास की गुणवत्ता जान सकते हैं और समझ सकते हैं कि किस आसन को और बेहतर करने की आवश्यकता है।  
यह टूल आपके योगाभ्यास को और अधिक प्रभावी, वैज्ञानिक तथा स्वयं-आकलन योग्य बनाता है।
.
""")

# Footer
st.markdown(
    f"""
    <div style='text-align:center; margin-top:40px; padding:15px; color:{text}; opacity:0.8;'>
        Developed by <b>Acharya Bhaskar</b> | © 2025 <br>
        <i>योग करें • स्वस्थ रहें</i>
    </div>
    """,
    unsafe_allow_html=True
)
