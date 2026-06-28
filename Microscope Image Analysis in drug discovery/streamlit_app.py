
import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Microscope AI — Drug Discovery Profiler",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .hero-title { font-family: 'Courier New', monospace; font-size: 2.2rem; font-weight: 700; color: #58a6ff; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
    .hero-sub { font-size: 0.95rem; color: #8b949e; margin-bottom: 2rem; font-family: monospace; }
    .section-label { font-family: monospace; font-size: 0.75rem; letter-spacing: 0.12em; color: #58a6ff; text-transform: uppercase; margin-bottom: 0.4rem; }
    .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.2rem; text-align: center; }
    .stat-value { font-size: 2rem; font-weight: 700; font-family: monospace; color: #58a6ff; }
    .stat-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
    .uploader-label { font-family: monospace; font-size: 0.8rem; color: #8b949e; margin-bottom: 0.3rem; }
    .status-running { display: inline-block; width: 8px; height: 8px; background: #3fb950; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .result-header { font-family: monospace; font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; padding-bottom: 0.5rem; border-bottom: 1px solid #21262d; margin-bottom: 1rem; }
    div[data-testid="stFileUploader"] { background: #161b22; border: 1px dashed #30363d; border-radius: 8px; padding: 0.5rem; }
    hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔬 Microscope AI — Cell Segmentation Profiler</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Human Protein Atlas · UNet ResNet34 · 4-Channel Fluorescence Microscopy · Drug Discovery Pipeline</div>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_session():
    try:
        return ort.InferenceSession("microscope_unet_256.onnx", providers=['CPUExecutionProvider'])
    except Exception:
        return None

session = load_session()

if session is not None:
    st.markdown('<span class="status-running"></span><span style="font-family:monospace;font-size:0.8rem;color:#3fb950;">ONNX Inference Engine — Online</span>', unsafe_allow_html=True)
else:
    st.error("⚠️ ONNX model failed to load.")

st.markdown("<br>", unsafe_allow_html=True)

def apply_clahe_per_channel(img_fused):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = np.zeros_like(img_fused)
    for c in range(img_fused.shape[2]):
        ch = img_fused[:, :, c]
        if ch.dtype != np.uint8:
            ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        equalized[:, :, c] = clahe.apply(ch)
    return equalized

def min_max_normalize(img_fused):
    img_fused = img_fused.astype(np.float32)
    for c in range(img_fused.shape[2]):
        min_val = img_fused[:, :, c].min()
        max_val = img_fused[:, :, c].max()
        if (max_val - min_val) > 0:
            img_fused[:, :, c] = (img_fused[:, :, c] - min_val) / (max_val - min_val)
        else:
            img_fused[:, :, c] = 0.0
    return img_fused

st.markdown('<div class="section-label">Step 1 — Upload 4 Fluorescence Channel Images</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:0.8rem;color:#8b949e;margin-bottom:1rem;">Each image is a separate grayscale PNG from the same microscopy acquisition. Upload all 4 to run inference.</div>', unsafe_allow_html=True)

cols = st.columns(4)
labels = ["🔵 Blue — Nuclei", "🔴 Red — Microtubules", "🟡 Yellow — ER", "🟢 Green — Target Protein"]
keys   = ["blue", "red", "yellow", "green"]
uploaded = {}
for col, label, key in zip(cols, labels, keys):
    with col:
        st.markdown(f'<div class="uploader-label">{label}</div>', unsafe_allow_html=True)
        uploaded[key] = st.file_uploader("", type=["png", "jpg"], key=key, label_visibility="collapsed")

all_uploaded = all(uploaded[k] is not None for k in keys)
n_uploaded = sum(1 for k in keys if uploaded[k] is not None)

if not all_uploaded:
    st.markdown(f'<div style="font-family:monospace;font-size:0.8rem;color:#8b949e;margin-top:0.5rem;">{n_uploaded}/4 channels uploaded — waiting for all 4 to run inference</div>', unsafe_allow_html=True)

st.markdown("---")

if all_uploaded and session is not None:
    with st.spinner("Running segmentation pipeline..."):
        channels = []
        for key in keys:
            raw = np.frombuffer(uploaded[key].read(), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            channels.append(cv2.resize(img, (256, 256)))

        fused = np.stack(channels, axis=-1)
        fused = min_max_normalize(apply_clahe_per_channel(fused))
        tensor = np.expand_dims(np.transpose(fused, (2, 0, 1)), axis=0).astype(np.float32)

        out = session.run(None, {session.get_inputs()[0].name: tensor})
        mask = np.argmax(out[0][0], axis=0)

        total_px = mask.size
        bg_pct   = int(np.sum(mask == 0)) / total_px * 100
        cell_pct = int(np.sum(mask == 1)) / total_px * 100
        nuc_pct  = int(np.sum(mask == 2)) / total_px * 100
        nuc_px   = int(np.sum(mask == 2))
        cell_px  = int(np.sum(mask == 1))
        ratio    = (cell_px / nuc_px) if nuc_px > 0 else 0

    st.markdown('<div class="section-label">Segmentation Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl in zip([s1,s2,s3,s4],
                              [f"{cell_pct:.1f}%", f"{nuc_pct:.1f}%", f"{bg_pct:.1f}%", f"{ratio:.2f}x"],
                              ["Cell Body Coverage","Nucleus Coverage","Background","Cell / Nucleus Ratio"]):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 2 — Results</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown('<div class="result-header">Composite Channel Overlay</div>', unsafe_allow_html=True)
        st.image(np.clip(fused[:, :, [1, 3, 0]], 0, 1), use_container_width=True)
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;font-family:monospace;">R: Microtubules · G: Target Protein · B: Nuclei</div>', unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="result-header">Predicted Segmentation Mask</div>', unsafe_allow_html=True)
        COLOR_MAP = {0: [13,17,23], 1: [56,189,248], 2: [249,115,22]}
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in COLOR_MAP.items():
            mask_rgb[mask == cls] = color
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.imshow(mask_rgb)
        ax.axis('off')
        patches = [
            mpatches.Patch(color=np.array(COLOR_MAP[0])/255, label=f'Background ({bg_pct:.1f}%)'),
            mpatches.Patch(color=np.array(COLOR_MAP[1])/255, label=f'Cell Body ({cell_pct:.1f}%)'),
            mpatches.Patch(color=np.array(COLOR_MAP[2])/255, label=f'Nucleus ({nuc_pct:.1f}%)'),
        ]
        ax.legend(handles=patches, loc='lower left', fontsize=9, framealpha=0.85,
                  facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Individual Channel Inspection</div>', unsafe_allow_html=True)
    ch_cols = st.columns(4)
    ch_labels = ["Blue · Nuclei","Red · Microtubules","Yellow · ER","Green · Target Protein"]
    ch_cmaps  = ["Blues","Reds","YlOrBr","Greens"]
    for i, (col, lbl, cmap) in enumerate(zip(ch_cols, ch_labels, ch_cmaps)):
        with col:
            fig2, ax2 = plt.subplots(figsize=(3,3))
            fig2.patch.set_facecolor('#0d1117')
            ax2.set_facecolor('#0d1117')
            ax2.imshow(fused[:,:,i], cmap=cmap)
            ax2.set_title(lbl, fontsize=7, color='#8b949e', fontfamily='monospace', pad=4)
            ax2.axis('off')
            st.pyplot(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div style="font-family:monospace;font-size:0.75rem;color:#484f58;text-align:center;">DEPI Graduation Project · Microscope AI Segmentation & Deployment · Human Protein Atlas · UNet ResNet34</div>', unsafe_allow_html=True)

elif all_uploaded and session is None:
    st.error("⚠️ All 4 channels uploaded but ONNX model is offline.")
