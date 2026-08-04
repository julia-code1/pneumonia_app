<div align="center">

# ◍ PNEUMA
### AI-Assisted Radiological Screening

A deep learning system that classifies chest X-rays as Normal or Pneumonia,
with Grad-CAM explainability — a CNN trained from scratch in TensorFlow/Keras,
served with Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-0B0E11?style=for-the-badge&logo=python&logoColor=5FD4E8)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-0B0E11?style=for-the-badge&logo=tensorflow&logoColor=5FD4E8)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-0B0E11?style=for-the-badge&logo=streamlit&logoColor=5FD4E8)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-0B0E11?style=for-the-badge&color=5FD4E8)](#license)

[Live Demo](#) · [Features](#features) · [Getting Started](#getting-started)

</div>

---

> ⚠ **Research and portfolio demonstration only.** PNEUMA is not FDA-cleared,
> has not been clinically validated, and must never be used for real
> diagnostic or treatment decisions. Always consult a licensed radiologist
> or physician. See [A Note on Accuracy](#model-performance) below.

## Overview

PNEUMA classifies chest X-rays as **Normal** or **Pneumonia** using a
convolutional neural network, and explains *why* via **Grad-CAM** — a
heatmap overlay showing which regions of the scan most influenced each
prediction, not just a bare label.

<table>
<tr>
<td width="50%" valign="top">

**What it does**
- Classifies a chest X-ray as Normal or Pneumonia
- Shows a Grad-CAM heatmap explaining the prediction
- Works on bundled sample scans or your own uploaded image
- Reports real, honestly-calibrated confidence — not a hardcoded 0.5 cutoff
- Exports a branded PDF screening report

</td>
<td width="50%" valign="top">

**What makes it different**
- Trained **from scratch** — no pretrained ImageNet weights — and says so
- A calibrated decision threshold (tuned on validation data), not a naive 0.5
- Grad-CAM on every single prediction, not just a demo screenshot
- A "Radiology Reading Room" visual theme — genuinely distinct from a
  typical light medical-app look
- Transparent, prominent limitations — dataset size, source population,
  real test accuracy — instead of an inflated headline number

</td>
</tr>
</table>

---

## Features

### Diagnose
- Choose from bundled sample X-rays or upload your own (JPG/PNG)
- Instant classification with a confidence score
- **Grad-CAM overlay** — a heatmap showing which regions drove the prediction
- **PDF screening report** — branded, with the same medical disclaimer built in

### Model Performance
- Accuracy, sensitivity (recall), precision, and AUC on held-out test data
- Confusion matrix
- An honest breakdown of *why* the accuracy is what it is, and what a
  transfer-learning approach would likely improve

### About
- Model architecture details
- Responsible-use guidelines
- Dataset provenance

---

## Dataset

[**Chest X-Ray Images (Pneumonia)**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— Kaggle (Paul Mooney), originally from **Kermany et al., *Cell*, 2018**,
sourced from Guangzhou Women and Children's Medical Center.

A curated, resized subset is bundled at `data/chest_xray/` (1,200 training
images, balanced 600/600 Normal/Pneumonia; 160 test images) so training
works fully offline. A handful of higher-resolution sample scans are at
`data/samples/` for the in-app demo gallery.

**Why a subset, not the full ~5,800-image dataset?** To keep the repo small
and training fast (~3 minutes on CPU) for a portfolio/demo context. Point
`train_model.py` at the full dataset for a stronger model — see
[Refreshing the dataset](#refreshing-the-dataset).

---

## Model Performance

Held-out test set:

| Metric | Value |
|---|---|
| Accuracy | ~67–69% |
| Sensitivity (Recall) | ~72–74% |
| Precision | ~66–68% |
| AUC | ~0.78–0.81 |

**A note on accuracy, honestly.** This is meaningfully below the 90%+ that
published work on this exact dataset achieves — because those results
almost always use **transfer learning** (ImageNet-pretrained backbones like
MobileNet or ResNet) plus the full training set. This build trains a small
CNN **from scratch**, no pretrained weights, on a reduced subset — a
deliberate, disclosed tradeoff for a fast, fully self-contained demo, not
an attempt to hide a weaker number behind a stronger-looking one.

**On the decision threshold.** With this little training data, a fixed 0.5
cutoff on the model's sigmoid output repeatedly collapsed to predicting a
single class for every input, despite the model's raw scores carrying real
signal (AUC well above 0.5). `train_model.py` instead sweeps thresholds
against a validation split and picks the one maximizing F1 — that
calibrated threshold (not 0.5) is what the app actually uses, and it's
shown in the UI.

---

## Design System

PNEUMA uses a deep clinical "radiology reading room" theme — near-black,
like an actual dark room used to read film — with a cool cyan-white
"monitor glow" as the primary accent, and clinical semantic colors for
results: green for Normal, red for a Pneumonia flag.

<div align="center">

| Token | Swatch | Hex |
|---|---|---|
| Background | ![#0B0E11](https://placehold.co/60x20/0B0E11/0B0E11.png) | `#0B0E11` |
| Panel | ![#12161A](https://placehold.co/60x20/12161A/12161A.png) | `#12161A` |
| Border | ![#262E35](https://placehold.co/60x20/262E35/262E35.png) | `#262E35` |
| Monitor glow | ![#5FD4E8](https://placehold.co/60x20/5FD4E8/5FD4E8.png) | `#5FD4E8` |
| Normal (result) | ![#34D399](https://placehold.co/60x20/34D399/34D399.png) | `#34D399` |
| Pneumonia flag | ![#F0564A](https://placehold.co/60x20/F0564A/F0564A.png) | `#F0564A` |

</div>

**Typography** — IBM Plex Sans (clean, technical display headings) · Inter (body) · IBM Plex Mono (all numeric data)

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Model** | TensorFlow/Keras — custom CNN (3 conv blocks, global average pooling), trained from scratch |
| **Explainability** | Grad-CAM (gradient-weighted class activation mapping) |
| **App / UI** | Streamlit (segmented controls, file uploader, bordered containers) |
| **Visualization** | Plotly (confusion matrix heatmap) |
| **Reporting** | ReportLab (PDF generation) |
| **Data** | Pillow, NumPy, Pandas, scikit-learn (metrics) |

---

## Getting Started

### Prerequisites
- Python 3.10+
- ~2 GB free disk space (TensorFlow is the big one)

### Installation

```bash
git clone <your-repo-url>
cd pneuma

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

Takes roughly 2–4 minutes on CPU. Saves `model.keras` and `metrics.pkl`.
The app also auto-trains on first load if these are missing.

### Run

```bash
streamlit run app.py
```

First load takes longer than the other apps in this portfolio — TensorFlow
itself takes a few seconds to import, plus model load. Open the URL
Streamlit prints (usually `http://localhost:8501`).

### Troubleshooting

<details>
<summary><b>ModuleNotFoundError / package not found after install</b></summary>

Usually means `pip install -r requirements.txt` didn't finish. A common cause:

```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

TensorFlow is large — make sure you have at least 2 GB free, then:

```bash
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

If `C:` (not just the drive your project is on) is nearly full, redirect
pip's temp directory too:

```powershell
$env:TEMP = "D:\temp"
$env:TMP = "D:\temp"
mkdir D:\temp -Force
```

</details>

<details>
<summary><b>App seems stuck / blank on first load</b></summary>

This is normal for the first ~20–30 seconds — TensorFlow import plus model
training (if `model.keras` doesn't exist yet) takes longer than a typical
Streamlit app. Give it a minute before assuming something's wrong.

</details>

---

## Project Structure

```
pneuma/
├── app.py                       # Streamlit app (Diagnose, Performance, About)
├── train_model.py               # CNN training script
├── data/
│   ├── chest_xray/               # Curated train/test/val subset
│   │   ├── train/{NORMAL,PNEUMONIA}/
│   │   ├── test/{NORMAL,PNEUMONIA}/
│   │   └── val/{NORMAL,PNEUMONIA}/
│   └── samples/                  # Higher-res demo gallery images
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Dark theme base
├── .gitignore
├── LICENSE
└── README.md
```

`model.keras` and `metrics.pkl` are **not** committed — the app trains and
caches them automatically on first load.

---

## Deployment

### Streamlit Community Cloud

1. Push this folder (including `data/`) to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch, and set the main file to `app.py`.
4. Deploy — dependencies install from `requirements.txt` and the model
   trains automatically on first load (may take a few minutes on first boot).

### Refreshing the dataset

Download the [full dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
from Kaggle, replace the contents of `data/chest_xray/` keeping the same
`{split}/{NORMAL,PNEUMONIA}/` folder structure, delete any cached
`model.keras`, and rerun `python train_model.py`. Consider also switching
to transfer learning (e.g. MobileNetV2 with ImageNet weights) for a
meaningfully stronger model if internet access to download pretrained
weights is available in your environment.

---

## Responsible Use

- **Not a diagnostic device.** Not FDA-cleared or clinically validated.
- **Not trained on diverse populations.** The source dataset is from a
  single pediatric hospital in Guangzhou, China — performance on other age
  groups, scanners, or populations is unknown.
- **Small training set.** ~1,200 images, far fewer than a clinical-grade
  model would use.
- **Always consult a licensed radiologist or physician** for any real
  health concern.

---

## Roadmap

- [ ] Transfer learning (MobileNetV2/ResNet) for a stronger baseline
- [ ] Multi-class classification (bacterial vs. viral pneumonia)
- [ ] Train on the full ~5,800-image dataset
- [ ] Model calibration curve / reliability diagram in the Performance tab

---

## License

Code distributed under the MIT License — see [`LICENSE`](LICENSE). The
bundled dataset is subject to its own terms; see the Kaggle page linked
above for details.

## Author

**Hamna Munir**
Software Engineering & AI/ML

<p>
<a href="#"><img src="https://img.shields.io/badge/GitHub-0B0E11?style=for-the-badge&logo=github&logoColor=5FD4E8" /></a>
<a href="#"><img src="https://img.shields.io/badge/LinkedIn-0B0E11?style=for-the-badge&logo=linkedin&logoColor=5FD4E8" /></a>
</p>

<sub>Update the badge links above with your actual profile URLs.</sub>

---

<div align="center">
<sub>Built as a portfolio deep-learning project · Dataset © Kermany et al., Cell 2018 / Guangzhou Women and Children's Medical Center</sub>
</div>
