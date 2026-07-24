import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from cnn import CNN, BilinearCNN
import numpy as np
import io
import os
import glob
import timm


# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AerialVision · CSC3109",
    page_icon="🛰️",
    layout="wide",
)

# ─────────────────────────────────────────
# Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

/* Hero */
.hero {
    border-bottom: 1px solid #1e2130;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #4a9eff;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 300;
}

/* Panel */
.panel {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.panel-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #4a9eff;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Prediction result */
.pred-class {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
}
.pred-conf {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #4a9eff;
    margin-top: 0.2rem;
}

/* Confidence bar */
.bar-row { margin-bottom: 0.6rem; }
.bar-label {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-bottom: 0.2rem;
    display: flex;
    justify-content: space-between;
}
.bar-track {
    background: #1e2130;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}

/* Tag */
.model-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    background: #1e2130;
    color: #4a9eff;
    border: 1px solid #2a3050;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    letter-spacing: 0.05em;
}

/* Status dot */
.dot-ready { color: #22c55e; }
.dot-na    { color: #374151; }

/* Divider */
.divider { border: none; border-top: 1px solid #1e2130; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
CLASSES    = ["crosswalk", "intersection", "parking_lot", "parking_space"]
IMG_SIZE   = 224
MODEL_DIR  = "model"

# Bar accent colours per class
CLASS_COLORS = {
    "crosswalk":     "#4a9eff",
    "intersection":  "#a78bfa",
    "parking_lot":   "#34d399",
    "parking_space": "#fb923c",
}

# ─────────────────────────────────────────
# Model registry
# Each entry: display name, filename, builder fn
# Add teammates' models here as they finish
# ─────────────────────────────────────────
def build_resnet50(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, num_classes))
    return m

def build_efficientnet(num_classes):
    m = models.efficientnet_b3(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return m

def build_bilinear_cnn(num_classes):
    # pretrained=False / freeze_backbone=False: we're loading fully-trained
    # weights for inference, not initializing for training.
    return BilinearCNN(num_classes, pretrained=False, freeze_backbone=False)

def build_densenet(num_classes):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def build_vit(num_classes):
    m = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return m

def build_custom_cnn(num_classes):
    return CNN(num_classes=num_classes)

MODEL_REGISTRY = {
    "ResNet-50":    {"file": "resnet50_final.pt",    "builder": build_resnet50,   "ready": True},
    "EfficientNet-B3": {"file": "efficientnet_b3_latest.pt","builder": build_efficientnet,"ready": True},
    "Bilinear CNN":    {"file": "bilinear_cnn_final.pt",   "builder": build_bilinear_cnn,  "ready": True},
    "DenseNet121":     {"file": "densenet_model.pt",    "builder": build_densenet,   "ready": True},
    "Visual Transformer":  {"file": "vit_model.pt",  "builder": build_vit,        "ready": True},
    "Custom CNN":   {"file": "custom_cnn_model.pt",  "builder": build_custom_cnn, "ready": True},
}

# Auto-detect which models are actually available on disk
for name, cfg in MODEL_REGISTRY.items():
    path = os.path.join(MODEL_DIR, cfg["file"])
    cfg["available"] = os.path.exists(path)

# ─────────────────────────────────────────
# Transform (val pipeline — no augmentation)
# ─────────────────────────────────────────
MODEL_TRANSFORMS = {
    "Visual Transformer": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        # vit pretrained model works better with imagenet mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

# Default transform used by all other models (dataset-specific stats)
default_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3621, 0.3615, 0.3383],
                         std=[0.1419, 0.1417, 0.1363]),
])
# ─────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    cfg     = MODEL_REGISTRY[model_name]
    builder = cfg["builder"]
    path    = os.path.join(MODEL_DIR, cfg["file"])
    if builder is None:
        raise NotImplementedError(f"Builder for {model_name} not yet defined.")
    model = builder(len(CLASSES))
    # weights_only=False: these checkpoints bundle plain Python metadata
    # (classes, norm stats, etc.) alongside the tensors, which newer
    # torch versions won't unpickle under the weights_only=True default.
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and checkpoint dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────
def predict(model, img: Image.Image, model_name):
    transform = MODEL_TRANSFORMS.get(model_name, default_transform)
    tensor = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()
    top_idx  = int(np.argmax(probs))
    return CLASSES[top_idx], probs

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">CSC3109 · Group 14 · Aerial Image Classification</div>
  <div class="hero-title">AerialVision</div>
  <div class="hero-sub">Classify aerial road infrastructure — crosswalk, intersection, parking lot, parking space</div>
</div>
""", unsafe_allow_html=True)

# Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── Model selector ──
    st.markdown('<div class="panel-label">Model</div>', unsafe_allow_html=True)

    available_models = [n for n, c in MODEL_REGISTRY.items() if c["available"] and c["builder"]]
    unavailable      = [n for n, c in MODEL_REGISTRY.items() if not c["available"] or not c["builder"]]

    if not available_models:
        st.error("No model weights found in `model/` folder.")
        st.stop()

    selected_model = st.selectbox(
        "Select model",
        available_models,
        label_visibility="collapsed"
    )


    # Status grid
    status_html = ""
    for name in MODEL_REGISTRY:
        cfg  = MODEL_REGISTRY[name]
        dot  = "dot-ready" if cfg["available"] and cfg["builder"] else "dot-na"
        icon = "●" if cfg["available"] and cfg["builder"] else "○"
        status_html += f'<span class="{dot}" style="margin-right:1rem;font-size:0.8rem">{icon} {name}</span>'
    st.markdown(f'<div style="margin-top:0.8rem">{status_html}</div>', unsafe_allow_html=True)

    # ── Input mode ──
    st.markdown('<div class="panel-label">Input</div>', unsafe_allow_html=True)

    mode = st.radio("Input mode", ["Upload image", "Use webcam"], label_visibility="collapsed")

    image = None
    if mode == "Upload image":
        uploaded = st.file_uploader("Drop an image", type=["jpg","jpeg","png"],
                                     label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded)
    else:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam:
            image = Image.open(cam)

with col_right:
    # Use Streamlit's built-in container to get a clean, properly sized box
    with st.container(border=True):
        st.markdown('<div class="panel-label">Result</div>', unsafe_allow_html=True)

        if image is None:
            st.markdown('<p style="color:#4b5563;font-size:0.9rem;margin-top:3rem;text-align:center">Upload an image or take a photo to classify</p>', unsafe_allow_html=True)
        else:
            st.image(image, width=400)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            with st.spinner("Classifying…"):
                try:
                    model = load_model(selected_model)
                    pred_class, probs = predict(model, image, selected_model)

                    # Predicted class
                    conf = max(probs) * 100
                    st.markdown(f"""
                    <div class="pred-class">{pred_class.replace("_", " ").title()}</div>
                    <div class="pred-conf">{conf:.1f}% confidence · <span class="model-tag">{selected_model}</span></div>
                    """, unsafe_allow_html=True)

                    st.markdown('<hr class="divider">', unsafe_allow_html=True)

                    # Confidence bars
                    st.markdown('<div class="panel-label">All classes</div>', unsafe_allow_html=True)
                    for cls, prob in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
                        pct   = prob * 100
                        color = CLASS_COLORS.get(cls, "#4a9eff")
                        st.markdown(f"""
                        <div class="bar-row">
                            <div class="bar-label">
                                <span>{cls.replace("_"," ").title()}</span>
                                <span style="font-family:'Space Mono',monospace;font-size:0.75rem">{pct:.1f}%</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading model: {e}")