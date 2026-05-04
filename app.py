import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import sys
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liver & Tumor Segmentation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS (premium dark-mode aesthetic) ──────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* ── Card wrapper ── */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }

    /* ── Hero header ── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
    }
    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.55);
        font-size: 1rem;
        margin-bottom: 32px;
    }

    /* ── Metric badges ── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px 4px 0 0;
    }
    .badge-bg   { background: rgba(100,116,139,0.35);  color: #94a3b8; }
    .badge-liver{ background: rgba(96,165,250,0.25);   color: #60a5fa; }
    .badge-tumor{ background: rgba(248,113,113,0.25);  color: #f87171; }

    /* ── Uploader area ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(167,139,250,0.5) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.03) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* ── Info / warning boxes ── */
    .stAlert { border-radius: 10px !important; }

    /* ── Metric labels ── */
    [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; }
    [data-testid="stMetricValue"]  { color: #a78bfa !important; font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_unet.pth")

# ── Colour map for the 3 classes ─────────────────────────────────────────────
CLASS_COLORS = np.array([
    [15,  15,  30 ],   # 0 – background  (near-black)
    [96,  165, 250],   # 1 – liver       (blue)
    [248, 113, 113],   # 2 – tumour      (red)
], dtype=np.uint8)

CLASS_LABELS = {0: "Background", 1: "Liver", 2: "Tumor"}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    # Import the UNet definition from model.py in the same directory
    sys.path.insert(0, os.path.dirname(__file__))
    from model import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=3)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(model, device, pil_image: Image.Image):
    """
    Accepts a PIL image (any mode), converts to grayscale float32 tensor,
    runs U-Net, and returns:
      - pred_mask  : (H, W) numpy int64 array with class indices 0/1/2
      - colour_mask: (H, W, 3) RGB numpy uint8 array
    """
    img_gray = pil_image.convert("L")
    H, W = img_gray.size[1], img_gray.size[0]

    arr = np.array(img_gray, dtype=np.float32) / 255.0          # (H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                                   # (1, 3, H, W)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W)

    colour = CLASS_COLORS[pred]                                  # (H, W, 3)
    return pred, colour


# ── Pixel-count statistics ────────────────────────────────────────────────────
def compute_stats(pred_mask):
    total = pred_mask.size
    counts = {cls: int((pred_mask == cls).sum()) for cls in [0, 1, 2]}
    pcts   = {cls: counts[cls] / total * 100       for cls in [0, 1, 2]}
    return counts, pcts


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 Liver & Tumor Segmentation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a CT scan slice · U-Net automatically detects Liver and Tumor regions</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    show_overlay = st.toggle("Show overlay on CT", value=True)
    overlay_alpha = st.slider("Overlay opacity", 0.1, 1.0, 0.55, 0.05,
                              disabled=not show_overlay)
    show_colorbar = st.toggle("Show colour legend", value=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        """
        **Architecture**: U-Net (5 encoder / 5 decoder levels)  
        **Dataset**: LiTS (Liver Tumour Segmentation)  
        **Input**: Grayscale CT slice (any resolution)  
        **Classes**:  
        &nbsp;&nbsp;⬛ `0` Background  
        &nbsp;&nbsp;🔵 `1` Liver  
        &nbsp;&nbsp;🔴 `2` Tumour  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.35)'>Model: best_unet.pth</small>",
        unsafe_allow_html=True,
    )

# ── Load model (spinner shown once) ──────────────────────────────────────────
with st.spinner("🔄 Loading U-Net model…"):
    try:
        model, device = load_model()
        st.success(f"✅ Model loaded  ·  Device: **{device}**", icon="🚀")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown("---")
uploaded = st.file_uploader(
    "📂 Upload a CT scan slice (PNG / JPG / TIFF / BMP)",
    type=["png", "jpg", "jpeg", "tiff", "bmp"],
    help="Any standard image format is accepted. The model converts it to grayscale internally.",
)

if uploaded is None:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:48px 24px;">
            <div style="font-size:3rem">🖼️</div>
            <div style="font-size:1.1rem; color:rgba(255,255,255,0.5); margin-top:12px;">
                Upload a CT scan slice above to begin segmentation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Run inference ─────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded)

with st.spinner("⚡ Running segmentation…"):
    pred_mask, colour_mask = run_inference(model, device, pil_img)

counts, pcts = compute_stats(pred_mask)

# ── Layout: metrics row ───────────────────────────────────────────────────────
st.markdown("### 📊 Segmentation Results")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("🖤 Background", f"{pcts[0]:.1f}%", f"{counts[0]:,} px")
with col_m2:
    st.metric("🔵 Liver", f"{pcts[1]:.1f}%", f"{counts[1]:,} px")
with col_m3:
    st.metric("🔴 Tumor", f"{pcts[2]:.1f}%", f"{counts[2]:,} px")

st.markdown("---")

# ── Layout: image columns ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

# ── 1. Original CT ────────────────────────────────────────────────────────────
with col1:
    st.markdown("**Original CT Slice**")
    gray_img = pil_img.convert("L")
    st.image(gray_img, use_container_width=True, clamp=True)

# ── 2. Segmentation mask ──────────────────────────────────────────────────────
with col2:
    st.markdown("**Segmentation Mask**")
    seg_pil = Image.fromarray(colour_mask, mode="RGB")
    st.image(seg_pil, use_container_width=True)

# ── 3. Overlay / legend ───────────────────────────────────────────────────────
with col3:
    if show_overlay:
        st.markdown("**CT + Mask Overlay**")
        # Convert grayscale to RGB so we can alpha-blend
        gray_rgb = np.stack([np.array(gray_img)] * 3, axis=-1)   # (H, W, 3)
        blended  = (gray_rgb * (1 - overlay_alpha) +
                    colour_mask * overlay_alpha).astype(np.uint8)
        st.image(blended, use_container_width=True)
    else:
        st.markdown("&nbsp;")

# ── Colour legend (matplotlib) ────────────────────────────────────────────────
if show_colorbar:
    st.markdown("---")
    st.markdown("**Legend**")
    badge_html = (
        '<span class="badge badge-bg">⬛ Background</span>'
        '<span class="badge badge-liver">🔵 Liver</span>'
        '<span class="badge badge-tumor">🔴 Tumor</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)

# ── Download section ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💾 Download Results")

dl1, dl2, dl3 = st.columns(3)

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

with dl1:
    st.download_button(
        "⬇️ Download Mask (PNG)",
        data=pil_to_bytes(Image.fromarray(colour_mask)),
        file_name="segmentation_mask.png",
        mime="image/png",
        use_container_width=True,
    )

with dl2:
    # Raw class-index mask (0/1/2) scaled to 0/127/255 for viewing
    raw_vis = (pred_mask * 127).astype(np.uint8)
    st.download_button(
        "⬇️ Download Raw Labels (PNG)",
        data=pil_to_bytes(Image.fromarray(raw_vis, mode="L")),
        file_name="segmentation_labels.png",
        mime="image/png",
        use_container_width=True,
    )

with dl3:
    if show_overlay:
        blended_pil = Image.fromarray(blended)
    else:
        blended_pil = Image.fromarray(
            (np.stack([np.array(gray_img)] * 3, axis=-1) * 0.5 +
             colour_mask * 0.5).astype(np.uint8)
        )
    st.download_button(
        "⬇️ Download Overlay (PNG)",
        data=pil_to_bytes(blended_pil),
        file_name="overlay.png",
        mime="image/png",
        use_container_width=True,
    )

# ── Full-width matplotlib figure (optional detail view) ──────────────────────
st.markdown("---")
with st.expander("🔬 Detailed Analysis View", expanded=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             facecolor="#0f0c29")
    titles   = ["CT Scan (Gray)", "Segmentation Mask", "Overlay"]
    imgs     = [
        np.array(gray_img),
        colour_mask,
        (np.stack([np.array(gray_img)] * 3, axis=-1) * 0.5 + colour_mask * 0.5).astype(np.uint8),
    ]
    cmaps    = ["gray", None, None]

    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.set_title(title, color="white", fontsize=13, pad=8)
        ax.axis("off")
        ax.set_facecolor("#0f0c29")
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)

    # Legend
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=CLASS_LABELS[i])
        for i in range(3)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               framealpha=0.2, labelcolor="white", fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    st.pyplot(fig)
    plt.close(fig)

st.markdown(
    "<div style='text-align:center;color:rgba(255,255,255,0.2);font-size:0.8rem;margin-top:32px;'>"
    "U-Net Liver &amp; Tumor Segmentation · LiTS Dataset · PyTorch"
    "</div>",
    unsafe_allow_html=True,
)
