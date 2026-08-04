import io
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import tensorflow as tf
from tensorflow import keras

# ==============================================================================
# Page config
# ==============================================================================
st.set_page_config(
    page_title="PNEUMA — AI-Assisted Radiological Screening",
    page_icon="◍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# Design tokens + global CSS — deep clinical "radiology reading room" theme
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root{
    --bg:#0B0E11;
    --panel:#12161A;
    --panel-alt:#171C21;
    --border:#262E35;
    --border-soft:#1B2126;
    --text:#EDF1F4;
    --text-muted:#7A8894;
    --text-faint:#4A555E;
    --glow:#5FD4E8;
    --glow-dim:#2A8FA3;
    --normal:#34D399;
    --abnormal:#F0564A;
    --abnormal-dim:#7A2A24;
    --amber:#F0A83E;
}

html, body, .stApp { background: var(--bg) !important; }
* { font-family: 'Inter', -apple-system, sans-serif; }

#MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }

/* ---------- Header ---------- */
.clinic-header {
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 1.2rem; margin-bottom: 1.2rem; border-bottom: 1px solid var(--border);
}
.wordmark-row { display: flex; align-items: center; gap: 0.9rem; }
.wordmark {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 2rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.01em; line-height: 1;
}
.wordmark .dot { color: var(--glow); }
.tagline {
    font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.18em;
    text-transform: uppercase; margin-top: 0.35rem; font-weight: 600;
}
.header-meta { text-align: right; font-size: 0.72rem; color: var(--text-faint); font-family: 'IBM Plex Mono', monospace; line-height: 1.6; }

/* ---------- Disclaimer banner ---------- */
.disclaimer-banner {
    background: linear-gradient(90deg, rgba(240,168,62,0.10), rgba(240,168,62,0.03));
    border: 1px solid rgba(240,168,62,0.35); border-left: 3px solid var(--amber);
    border-radius: 6px; padding: 0.9rem 1.2rem; margin-bottom: 1.6rem;
    font-size: 0.82rem; color: var(--text); line-height: 1.55;
}
.disclaimer-banner b { color: var(--amber); }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 2.2rem; border-bottom: 1px solid var(--border-soft); }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--text-muted); font-weight: 600;
    font-family: 'IBM Plex Sans', sans-serif; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase;
    padding: 0 0 0.75rem 0;
}
.stTabs [aria-selected="true"] { color: var(--glow) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--glow) !important; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Section labels ---------- */
.section-eyebrow {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 1.02rem; font-weight: 600; color: var(--text);
    margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-eyebrow::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--glow); display: inline-block; }
.section-note { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1.1rem; }

/* ---------- Cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); }
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
    background: var(--panel); border-radius: 8px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div) {
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}

/* ---------- Diagnosis result ---------- */
.diagnosis-badge {
    display: inline-flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 1.1rem; border-radius: 6px; font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700; font-size: 1.1rem; letter-spacing: 0.03em; text-transform: uppercase;
}
.diagnosis-badge.normal { background: rgba(52,211,153,0.12); border: 1px solid var(--normal); color: var(--normal); }
.diagnosis-badge.abnormal { background: rgba(240,86,74,0.12); border: 1px solid var(--abnormal); color: var(--abnormal); }
.diagnosis-dot { width: 10px; height: 10px; border-radius: 50%; }
.diagnosis-dot.normal { background: var(--normal); box-shadow: 0 0 10px var(--normal); }
.diagnosis-dot.abnormal { background: var(--abnormal); box-shadow: 0 0 10px var(--abnormal); }

.confidence-label { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); font-weight: 700; margin-top: 1rem; }
.confidence-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.2rem; }

/* ---------- Sample gallery ---------- */
.sample-card {
    border-radius: 6px; overflow: hidden; border: 2px solid var(--border);
    cursor: pointer; transition: border-color 0.15s ease;
}
.sample-card.selected { border-color: var(--glow); }

/* ---------- KPI ---------- */
.kpi-label { font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem; font-weight: 700; color: var(--text); margin-top: 0.2rem; }
.kpi-sub { font-size: 0.7rem; margin-top: 0.15rem; font-weight: 600; color: var(--glow); }

/* ---------- Badges/tags ---------- */
.badge {
    display: inline-block; background: var(--panel-alt); border: 1px solid var(--border);
    color: var(--text); font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; padding: 0.3rem 0.75rem; border-radius: 4px; margin: 0.15rem;
}

/* ---------- Widgets ---------- */
[data-testid="stWidgetLabel"] p {
    font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-muted) !important; font-weight: 700 !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--panel-alt); border-color: var(--border); border-radius: 6px; color: var(--text);
}
[data-testid="stSegmentedControl"] label {
    background: var(--panel-alt) !important; border: 1px solid var(--border) !important;
    color: var(--text-muted) !important; border-radius: 6px !important;
}
[data-testid="stSegmentedControl"] label[data-checked="true"] {
    background: var(--glow) !important; color: #051018 !important; border-color: var(--glow) !important;
}

div.stButton > button {
    background: var(--glow); color: #051018; border: none; border-radius: 6px;
    padding: 0.72rem 1.4rem; font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; font-size: 0.82rem;
    letter-spacing: 0.04em; text-transform: uppercase; width: 100%;
}
div.stButton > button:hover { background: #7FE0EF; box-shadow: 0 6px 20px rgba(95,212,232,0.3); }

div[data-testid="stDownloadButton"] > button {
    background: transparent; color: var(--glow); border: 1px solid var(--glow-dim);
    border-radius: 6px; padding: 0.68rem 1.4rem; font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; font-size: 0.78rem;
    letter-spacing: 0.04em; text-transform: uppercase; width: 100%;
}
div[data-testid="stDownloadButton"] > button:hover { background: rgba(95,212,232,0.08); }

[data-testid="stFileUploader"] { background: var(--panel-alt); border: 1px dashed var(--border); border-radius: 8px; }
[data-testid="stMetricValue"] { color: var(--text); font-family: 'IBM Plex Mono', monospace; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }
[data-testid="stMetricDelta"] svg { display: none; }

hr { border-color: var(--border-soft); }
[data-testid="stDataFrame"] { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
::-webkit-scrollbar { height: 8px; width: 8px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    plot_bgcolor="#12161A",
    paper_bgcolor="#12161A",
    font=dict(color="#7A8894", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#262E35", zerolinecolor="#262E35"),
    yaxis=dict(gridcolor="#262E35", zerolinecolor="#262E35"),
    margin=dict(l=10, r=10, t=10, b=10),
)


def lung_icon_svg(size=44) -> str:
    """Original, simple geometric lung/respiratory icon — no external image."""
    return f"""
    <svg viewBox="0 0 64 64" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="lunggrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#5FD4E8"/>
                <stop offset="100%" stop-color="#2A8FA3"/>
            </linearGradient>
        </defs>
        <line x1="32" y1="8" x2="32" y2="26" stroke="url(#lunggrad)" stroke-width="3" stroke-linecap="round"/>
        <path d="M32 22 Q20 24 16 34 Q12 44 16 54 Q20 58 24 52 Q26 42 26 30 Q26 24 32 22Z" fill="url(#lunggrad)" opacity="0.85"/>
        <path d="M32 22 Q44 24 48 34 Q52 44 48 54 Q44 58 40 52 Q38 42 38 30 Q38 24 32 22Z" fill="url(#lunggrad)" opacity="0.85"/>
        <circle cx="32" cy="14" r="4" fill="none" stroke="#5FD4E8" stroke-width="2" opacity="0.6"/>
    </svg>
    """


# ==============================================================================
# Load model + metrics
# ==============================================================================
@st.cache_resource
def load_model():
    if not Path("model.keras").exists():
        from train_model import train_and_save
        train_and_save()
    model = keras.models.load_model("model.keras")
    metrics = pd.read_pickle("metrics.pkl") if Path("metrics.pkl").exists() else {}
    return model, metrics


model, metrics = load_model()
IMG_SIZE = metrics.get("img_size", 120)
THRESHOLD = metrics.get("threshold", 0.5)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img).astype("float32")
    return arr[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)


def predict(pil_img: Image.Image):
    x = preprocess_image(pil_img)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "PNEUMONIA" if prob >= THRESHOLD else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else (1 - prob)
    return label, prob, confidence, x


def grad_cam(x: np.ndarray, pred_index: int = 0) -> np.ndarray:
    """Grad-CAM heatmap over the 'last_conv' layer, showing which regions of
    the X-ray most influenced this specific prediction."""
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer("last_conv").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGB")
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img)

    # coral/glow colormap: low=transparent-dark, high=bright coral-amber
    colored = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    colored[..., 0] = np.clip(heatmap_arr * 1.4, 0, 255)         # R
    colored[..., 1] = np.clip(heatmap_arr * 0.55, 0, 255)        # G
    colored[..., 2] = np.clip((255 - heatmap_arr) * 0.25, 0, 255)  # B

    base_arr = np.array(base).astype("float32")
    colored = colored.astype("float32")
    mask = (heatmap_arr[..., None] / 255.0) * alpha
    blended = base_arr * (1 - mask) + colored * mask
    return Image.fromarray(np.uint8(blended))


def build_pdf_report(filename: str, label: str, confidence: float, prob: float) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    glow = colors.HexColor("#1E7A8C")
    dark = colors.HexColor("#12161A")
    muted = colors.HexColor("#5C6B75")
    result_color = colors.HexColor("#B23A30") if label == "PNEUMONIA" else colors.HexColor("#1B8A63")

    title_style = ParagraphStyle("Title", parent=styles["Title"], textColor=dark, fontSize=22, spaceAfter=2)
    tagline_style = ParagraphStyle("Tagline", parent=styles["Normal"], textColor=muted, fontSize=9,
                                    spaceAfter=18, alignment=TA_CENTER)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=glow, fontSize=12, spaceBefore=14, spaceAfter=6)
    result_style = ParagraphStyle("Result", parent=styles["Title"], fontSize=24, textColor=result_color,
                                   alignment=TA_CENTER, spaceAfter=4)
    conf_style = ParagraphStyle("Conf", parent=styles["Normal"], fontSize=11, textColor=muted,
                                 alignment=TA_CENTER, spaceAfter=16)

    elements = [
        Paragraph("PNEUMA", title_style),
        Paragraph("AI-ASSISTED RADIOLOGICAL SCREENING REPORT", tagline_style),
        Paragraph(label, result_style),
        Paragraph(f"Confidence: {confidence*100:.1f}%  |  Raw model score: {prob:.3f}", conf_style),
        Paragraph("Scan Details", h2_style),
    ]

    rows = [
        ["Filename", filename],
        ["Classification", label],
        ["Confidence", f"{confidence*100:.1f}%"],
        ["Decision threshold", f"{THRESHOLD:.2f}"],
        ["Model", "Custom CNN (trained from scratch)"],
        ["Report generated", datetime.now().strftime("%B %d, %Y %H:%M")],
    ]
    table = Table(rows, colWidths=[2.2 * inch, 3.3 * inch])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), muted),
        ("TEXTCOLOR", (1, 0), (1, -1), dark),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE1E4")),
    ]))
    elements.append(table)

    elements.append(Spacer(1, 20))
    disclaimer = (
        "IMPORTANT: This report is generated by an AI research/portfolio model and is NOT a "
        "medical diagnosis. It has not been validated for clinical use, was trained on a small "
        f"dataset, and achieves roughly {metrics.get('accuracy', 0)*100:.0f}% test accuracy. Any "
        "clinical decision must be made by a qualified radiologist or physician using appropriate "
        "diagnostic tools. Do not use this output to make real medical decisions."
    )
    elements.append(Paragraph(disclaimer, ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#B23A30"), leading=12)))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')} · PNEUMA AI-Assisted Screening (Portfolio/Research Demo)",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5, textColor=muted, alignment=TA_CENTER)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# Header
# ==============================================================================
st.markdown(f"""
<div class="clinic-header">
    <div class="wordmark-row">
        {lung_icon_svg()}
        <div>
            <div class="wordmark">PNEUMA<span class="dot">.</span></div>
            <div class="tagline">AI-Assisted Radiological Screening</div>
        </div>
    </div>
    <div class="header-meta">
        CUSTOM CNN &nbsp;·&nbsp; {metrics.get('n_test', 0)} TEST SCANS<br>
        TEST ACCURACY {metrics.get('accuracy', 0)*100:.0f}% &nbsp;·&nbsp; AUC {metrics.get('auc', 0):.2f}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <b>⚠ Research &amp; portfolio demonstration only — not a medical device.</b>
    This tool is not FDA-cleared, has not been clinically validated, and must never be used
    to make real diagnostic or treatment decisions. Always consult a licensed radiologist or
    physician. See the Model Performance and About tabs for this model's real, honestly-reported
    accuracy and limitations.
</div>
""", unsafe_allow_html=True)

tab_diagnose, tab_performance, tab_about = st.tabs(["Diagnose", "Model Performance", "About"])

# ==============================================================================
# DIAGNOSE TAB
# ==============================================================================
SAMPLES_DIR = Path("data/samples")


@st.cache_data
def list_samples():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(SAMPLES_DIR.glob("*.jpg"))


with tab_diagnose:
    col_input, col_result = st.columns([1, 1.3], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Select a Chest X-Ray</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-note">Choose a sample scan, or upload your own image (JPG/PNG).</div>', unsafe_allow_html=True)

            input_mode = st.segmented_control("Input source", ["Sample Scans", "Upload Image"], default="Sample Scans")

            selected_image = None
            selected_name = None

            if (input_mode or "Sample Scans") == "Sample Scans":
                samples = list_samples()
                if samples:
                    labels = [s.stem.replace("_", " ").title() for s in samples]
                    choice = st.selectbox("Sample scan", labels, index=0)
                    idx = labels.index(choice)
                    selected_image = Image.open(samples[idx])
                    selected_name = samples[idx].name
                    st.image(selected_image, use_container_width=True, caption=choice)
                else:
                    st.caption("No sample images bundled.")
            else:
                uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
                if uploaded:
                    selected_image = Image.open(uploaded)
                    selected_name = uploaded.name
                    st.image(selected_image, use_container_width=True)

            st.write("")
            analyze_clicked = st.button("Analyze Scan", use_container_width=True, disabled=selected_image is None)

    with col_result:
        if selected_image is not None:
            label, prob, confidence, x_input = predict(selected_image)
            heatmap = grad_cam(x_input)
            overlay = overlay_heatmap(selected_image, heatmap)

            with st.container(border=True):
                st.markdown('<div class="section-eyebrow">Screening Result</div>', unsafe_allow_html=True)

                badge_class = "abnormal" if label == "PNEUMONIA" else "normal"
                dot_class = badge_class
                st.markdown(f"""
                <div class="diagnosis-badge {badge_class}">
                    <span class="diagnosis-dot {dot_class}"></span> {label}
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="confidence-label">Model Confidence</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-value">{confidence*100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(min(max(confidence, 0.0), 1.0))
                st.caption(f"Raw model score: {prob:.3f}  ·  Decision threshold: {THRESHOLD:.2f}")

                st.write("")
                pdf_bytes = build_pdf_report(selected_name or "scan.jpg", label, confidence, prob)
                st.download_button(
                    "Download Screening Report (PDF)", data=pdf_bytes,
                    file_name=f"PNEUMA_Report_{(selected_name or 'scan').split('.')[0]}.pdf",
                    mime="application/pdf", use_container_width=True,
                )

            st.write("")
            with st.container(border=True):
                st.markdown('<div class="section-eyebrow">Grad-CAM — Where the Model Looked</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-note">Highlighted regions influenced this prediction most — a basic transparency check, not a clinical annotation.</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(selected_image.convert("L"), use_container_width=True, caption="Original")
                with c2:
                    st.image(overlay, use_container_width=True, caption="Grad-CAM Overlay")
        else:
            with st.container(border=True):
                st.markdown('<div class="section-eyebrow">Screening Result</div>', unsafe_allow_html=True)
                st.caption("Select or upload a chest X-ray to see a result here.")

# ==============================================================================
# MODEL PERFORMANCE TAB
# ==============================================================================
with tab_performance:
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Test Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%", f"{metrics.get('n_test', 0)} test scans"),
        ("Sensitivity (Recall)", f"{metrics.get('recall', 0)*100:.1f}%", "Correctly flagged pneumonia"),
        ("Precision", f"{metrics.get('precision', 0)*100:.1f}%", "Flagged cases that were real"),
        ("AUC", f"{metrics.get('auc', 0):.3f}", "Overall discriminative power"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], kpis):
        with col, st.container(border=True):
            st.markdown(f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                        f'<div class="kpi-sub">{sub}</div>', unsafe_allow_html=True)

    st.write("")
    col_cm, col_notes = st.columns([1, 1.1], gap="large")
    with col_cm:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Confusion Matrix</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-note">On {metrics.get("n_test", 0)} held-out test scans.</div>', unsafe_allow_html=True)
            cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            z = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]  # reorder for TP top-left
            fig_cm = go.Figure(go.Heatmap(
                z=z, x=["Predicted Pneumonia", "Predicted Normal"], y=["Actual Pneumonia", "Actual Normal"],
                colorscale=[[0, "#171C21"], [1, "#5FD4E8"]],
                text=z, texttemplate="%{text}", textfont=dict(size=18, color="#0B0E11", family="IBM Plex Mono"),
                showscale=False,
            ))
            fig_cm.update_layout(**DARK_LAYOUT, height=320)
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    with col_notes:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Reading the Numbers Honestly</div>', unsafe_allow_html=True)
            train_counts = metrics.get("train_counts", {})
            st.markdown(textwrap.dedent(f"""\
            This model was trained **from scratch** — no pretrained ImageNet
            weights — on a balanced subset of **{sum(train_counts.values())} chest X-rays**
            ({train_counts.get('NORMAL', 0)} normal, {train_counts.get('PNEUMONIA', 0)} pneumonia),
            drawn from the real [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
            dataset (Kermany et al., 2018).

            **Why not use a pretrained model?** Transfer learning (e.g.
            ImageNet-pretrained MobileNet/ResNet) is the standard, stronger
            approach for medical imaging with limited data — it typically
            pushes accuracy well above 90% on this exact dataset. This build
            trains from scratch instead, which is honestly reflected in the
            more modest **{metrics.get('accuracy', 0)*100:.0f}% accuracy** above.

            **The decision threshold ({THRESHOLD:.2f}, not 0.5)** was tuned on
            a validation split to maximize F1 — with limited training data, a
            fixed 0.5 cutoff repeatedly collapsed to predicting a single class.
            This is disclosed rather than hidden.
            """))

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Training Configuration</div>', unsafe_allow_html=True)
        cfg_badges = "".join(f'<span class="badge">{t}</span>' for t in [
            f"{metrics.get('epochs_run', 0)} epochs",
            f"{metrics.get('img_size', 0)}×{metrics.get('img_size', 0)} grayscale",
            f"{metrics.get('train_time_sec', 0):.0f}s train time",
            "CNN from scratch",
            "No pretrained weights",
        ])
        st.markdown(f'<div>{cfg_badges}</div>', unsafe_allow_html=True)

# ==============================================================================
# ABOUT TAB
# ==============================================================================
with tab_about:
    col_a, col_b = st.columns([1.2, 1], gap="large")
    with col_a:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">About PNEUMA</div>', unsafe_allow_html=True)
            st.markdown(textwrap.dedent("""\
            PNEUMA classifies chest X-rays as **Normal** or **Pneumonia** using a
            convolutional neural network trained from scratch in TensorFlow/Keras,
            with **Grad-CAM** explainability so every prediction ships with a visual
            explanation of which regions of the scan influenced it — not just a
            bare label.

            Built on the real [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
            dataset (Kermany et al., *Cell*, 2018), sourced from Guangzhou Women
            and Children's Medical Center.
            """))
            st.write("")
            st.markdown("**Model architecture**")
            st.markdown(textwrap.dedent("""\
            - 3 convolutional blocks (16 → 32 → 64 filters), each followed by
              max-pooling and dropout
            - No batch normalization — with this little training data, batch
              norm statistics proved unstable and repeatedly caused the model
              to collapse to predicting a single class
            - Global average pooling + a small dense head (32 units)
            - Data augmentation (flip, rotation, zoom) during training
            """))

    with col_b:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Responsible Use</div>', unsafe_allow_html=True)
            st.markdown(textwrap.dedent("""\
            - **Not a diagnostic device.** Not FDA-cleared or clinically validated.
            - **Not trained on diverse populations.** The source dataset is from
              a single pediatric hospital in Guangzhou, China — performance on
              other age groups, scanners, or populations is unknown.
            - **Small training set.** ~1,200 images total, far fewer than a
              clinical-grade model would use.
            - **Always consult a licensed radiologist or physician** for any
              real health concern.
            """))

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">Tech Stack</div>', unsafe_allow_html=True)
            stack_tags = "".join(f'<span class="badge">{t}</span>' for t in
                                  ["TensorFlow/Keras", "Grad-CAM", "Streamlit", "Plotly", "ReportLab", "Pillow"])
            st.markdown(f'<div>{stack_tags}</div>', unsafe_allow_html=True)

st.write("")
st.markdown("""
<div style="border-top: 1px solid #1B2126; margin-top: 2rem; padding-top: 1.4rem;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 0.6rem;">
    <div style="font-size: 0.72rem; color: #4A555E;">
        Developed by <span style="color:#7A8894; font-weight:600;">Hamna Munir</span>
        &nbsp;·&nbsp; Software Engineering &amp; AI/ML
    </div>
    <div style="font-size: 0.68rem; color: #4A555E; letter-spacing: 0.04em;">
        PNEUMA &nbsp;·&nbsp; BUILT WITH TENSORFLOW + STREAMLIT
    </div>
</div>
""", unsafe_allow_html=True)
