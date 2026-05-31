import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import uuid
from PIL import Image as PILImage

from predict import load_all_models, predict_image, predict_audio, generate_gradcam, get_region_activations, MODELS
from utils.image_utils import detect_face, overlay_heatmap, get_suspicious_regions
from utils.video_utils import analyze_video
from utils.audio_utils import extract_mel_spectrogram, get_frequency_band_activations
from utils.db_manager import init_db, save_scan, get_all_scans
from utils.pdf_generator import generate_pdf, generate_batch_summary_pdf

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Deepfake AI Detection", page_icon="🧠", layout="wide")
os.makedirs('temp', exist_ok=True)
os.makedirs('reports', exist_ok=True)
init_db()
load_all_models()

# ---------------- CSS (ONLY THIS SECTION CHANGED) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
.sidebar-brand {
    padding: 16px 0 24px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}
.sidebar-brand-title {
    font-size: 18px;
    font-weight: 700;
    color: #58a6ff !important;
    line-height: 1.3;
}
.sidebar-nav-label {
    font-size: 11px;
    color: #8b949e !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ---- HEADER ---- */
.header-wrap {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 32px 0 8px;
    margin-bottom: 8px;
}
.header-icon {
    font-size: 52px;
    line-height: 1;
}
.header-title {
    font-size: 40px;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
    margin: 0;
}
.header-subtitle {
    font-size: 15px;
    color: #8b949e;
    margin-top: 4px;
}

/* ---- HOME CARDS ---- */
.home-cards-wrap {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 32px 0 16px;
}
.home-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 28px 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
    cursor: pointer;
}
.home-card:hover {
    border-color: #58a6ff;
}
.home-card-icon {
    font-size: 48px;
    margin-bottom: 16px;
    display: block;
}
.home-card-title {
    font-size: 18px;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 8px;
}
.home-card-desc {
    font-size: 13px;
    color: #8b949e;
    margin-bottom: 20px;
    line-height: 1.5;
}
.home-card-btn {
    display: inline-block;
    background: #1f6feb;
    color: white !important;
    padding: 8px 24px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    border: none;
    cursor: pointer;
}
.home-card-btn:hover {
    background: #388bfd;
}

/* ---- SECTION HEADER ---- */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
    padding-bottom: 12px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}

/* ---- RESULT CARD ---- */
.result-wrap {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}

/* ---- BADGES ---- */
.fake-badge {
    display: inline-block;
    background: #da3633;
    color: white;
    padding: 6px 20px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 2px;
}
.real-badge {
    display: inline-block;
    background: #238636;
    color: white;
    padding: 6px 20px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 2px;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stButton > button:hover {
    background: #388bfd !important;
}
.stDownloadButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* ---- METRICS ---- */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* ---- DIVIDER ---- */
hr {
    border-color: #21262d !important;
}

/* ---- DATAFRAME ---- */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}

/* ---- PROGRESS BAR ---- */
.stProgress > div > div {
    background: #1f6feb !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (ONLY THIS SECTION CHANGED) ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🧠 Deepfake<br>AI Detection</div>
    </div>
    <div class="sidebar-nav-label">Navigation</div>
    """, unsafe_allow_html=True)
    page = st.radio("", ["Image Scan", "Video Scan", "Audio Scan", "Dashboard"],
                    label_visibility="collapsed")
    st.markdown("---")
    img_ok = MODELS['image'] is not None
    aud_ok = MODELS['audio'] is not None
    st.markdown(f"""
    <div style="font-size:11px;color:#8b949e;padding:4px 0;">
        <div style="margin-bottom:6px;">
            <span style="color:#{'3fb950' if img_ok else 'f85149'}">&#9679;</span>
            IMAGE: {'LOADED' if img_ok else 'NOT LOADED'}
        </div>
        <div>
            <span style="color:#{'3fb950' if aud_ok else 'f85149'}">&#9679;</span>
            AUDIO: {'LOADED' if aud_ok else 'NOT LOADED'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- HEADER (ONLY THIS SECTION CHANGED) ----------------
st.markdown("""
<div class="header-wrap">
    <div class="header-icon">🧠</div>
    <div>
        <div class="header-title">Deepfake AI Detection</div>
        <div class="header-subtitle">Multimodal Deepfake Detection System</div>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)

# ---------------- MODEL WARNING ----------------
if MODELS['image'] is None or MODELS['audio'] is None:
    st.warning("⚠ Model not loaded. Running in simulation mode.")

# ---------------- HOME PAGE CARDS (NEW - shown only on home) ----------------
if page == "Image Scan" and 'scan_started' not in st.session_state:
    st.markdown("""
    <div class="home-cards-wrap">
        <div class="home-card">
            <span class="home-card-icon">🖼️</span>
            <div class="home-card-title">Image Scan</div>
            <div class="home-card-desc">Upload and analyze suspicious images</div>
        </div>
        <div class="home-card">
            <span class="home-card-icon">🎥</span>
            <div class="home-card-title">Video Scan</div>
            <div class="home-card-desc">Upload and detect manipulated videos</div>
        </div>
        <div class="home-card">
            <span class="home-card-icon">🎧</span>
            <div class="home-card-title">Audio Scan</div>
            <div class="home-card-desc">Analyze voice recordings for tampering</div>
        </div>
        <div class="home-card">
            <span class="home-card-icon">📊</span>
            <div class="home-card-title">Dashboard</div>
            <div class="home-card-desc">View detailed scan reports and stats</div>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

# ================= IMAGE SCAN (NO LOGIC CHANGES) =================
if page == "Image Scan":
    st.markdown('<div class="section-title">🖼️ Image Scan</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload Image(s)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 1:
            st.info("Batch Scan Mode Activated")
            batch_results = []
            cols = st.columns(3)

            for i, file in enumerate(uploaded_files):
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                tmp_path = f"temp/{file.name}"
                cv2.imwrite(tmp_path, img)

                label, conf = predict_image(tmp_path)
                batch_results.append({'name': file.name, 'label': label, 'conf': round(conf * 100, 2)})

                with cols[i % 3]:
                    st.image(img_rgb, caption=f"{file.name} - {label}", use_container_width=True)

            df = pd.DataFrame(batch_results)
            st.dataframe(df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(df))
            c2.metric("Fake", len(df[df['label'] == 'FAKE']))
            c3.metric("Avg Conf", f"{df['conf'].mean():.1f}%")

        else:
            file = uploaded_files[0]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            tmp_path = f"temp/{file.name}"
            cv2.imwrite(tmp_path, img)

            face = detect_face(img_rgb)
            label, conf = predict_image(tmp_path)

            col1, col2 = st.columns(2)
            col1.image(face, caption="Detected Face", use_container_width=True)

            with col2:
                st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
                if label == "FAKE":
                    st.markdown("<span class='fake-badge'>FAKE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='real-badge'>REAL</span>", unsafe_allow_html=True)
                st.metric("Confidence", f"{conf:.2%}")
                st.progress(float(conf))
                st.markdown('</div>', unsafe_allow_html=True)

# ================= VIDEO SCAN (NO LOGIC CHANGES) =================
elif page == "Video Scan":
    st.markdown('<div class="section-title">🎥 Video Scan</div>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

    if uploaded_video:
        tfile = os.path.join("temp", uploaded_video.name)
        with open(tfile, "wb") as f:
            f.write(uploaded_video.read())

        st.video(tfile)

        if st.button("Analyze Video"):
            with st.spinner("Processing..."):
                results = analyze_video(tfile, predict_image)
                df = pd.DataFrame(results)

                fake_count = len(df[df['label'] == 'FAKE'])
                total = len(df)
                res = "FAKE" if fake_count / total > 0.3 else "REAL"

                st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
                if res == "FAKE":
                    st.markdown("<span class='fake-badge'>FAKE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='real-badge'>REAL</span>", unsafe_allow_html=True)
                st.write(f"Manipulated Frames: {fake_count}/{total}")
                st.markdown('</div>', unsafe_allow_html=True)

                fig, ax = plt.subplots(facecolor='#161b22')
                ax.set_facecolor('#161b22')
                sns.lineplot(data=df, x='frame', y='confidence', ax=ax, color='#58a6ff')
                ax.tick_params(colors='#8b949e')
                for sp in ax.spines.values():
                    sp.set_color('#21262d')
                plt.title("Confidence Timeline", color='#e6edf3')
                st.pyplot(fig)

# ================= AUDIO SCAN (NO LOGIC CHANGES) =================
elif page == "Audio Scan":
    st.markdown('<div class="section-title">🎧 Audio Scan</div>', unsafe_allow_html=True)

    uploaded_audio = st.file_uploader("Upload Audio", type=['wav', 'mp3'])

    if uploaded_audio:
        tfile = os.path.join("temp", uploaded_audio.name)
        with open(tfile, "wb") as f:
            f.write(uploaded_audio.read())

        st.audio(tfile)

        label, conf = predict_audio(tfile)

        st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
        if label == "FAKE":
            st.markdown("<span class='fake-badge'>FAKE</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='real-badge'>REAL</span>", unsafe_allow_html=True)
        st.metric("Confidence", f"{conf:.2%}")
        st.progress(float(conf))
        st.markdown('</div>', unsafe_allow_html=True)

# ================= DASHBOARD (NO LOGIC CHANGES) =================
elif page == "Dashboard":
    st.markdown('<div class="section-title">📊 Dashboard</div>', unsafe_allow_html=True)

    scans = get_all_scans()

    if scans:
        df = pd.DataFrame(scans, columns=['ID', 'ScanID', 'Type', 'File', 'Result', 'Conf', 'PDF', 'Timestamp'])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(df))
        c2.metric("Fake", len(df[df['Result'] == 'FAKE']))
        c3.metric("Real", len(df[df['Result'] == 'REAL']))

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            fig1, ax1 = plt.subplots(facecolor='#161b22')
            ax1.set_facecolor('#161b22')
            df['Result'].value_counts().plot.pie(
                autopct='%1.1f%%', ax=ax1,
                colors=['#f85149', '#3fb950'],
                textprops={'color': 'white'}
            )
            ax1.set_ylabel("")
            plt.title("Real vs Fake", color='#e6edf3')
            st.pyplot(fig1)

        with col_b:
            fig2, ax2 = plt.subplots(facecolor='#161b22')
            ax2.set_facecolor('#161b22')
            sns.countplot(data=df, x='Type', ax=ax2,
                         palette=['#1f6feb', '#388bfd', '#58a6ff'])
            ax2.tick_params(colors='#8b949e')
            for sp in ax2.spines.values():
                sp.set_color('#21262d')
            plt.title("Scans by Type", color='#e6edf3')
            st.pyplot(fig2)

        st.markdown("---")
        st.dataframe(df[['Timestamp', 'Type', 'File', 'Result', 'Conf']])
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;background:#161b22;border:1px solid #21262d;border-radius:12px;">
            <div style="font-size:48px;">📊</div>
            <div style="color:#8b949e;margin-top:12px;font-size:15px;">No scans recorded yet</div>
        </div>
        """, unsafe_allow_html=True)