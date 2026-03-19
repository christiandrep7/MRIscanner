# Brain Tumor Detection from MRI Scans — Project Plan

The goal of this project is to build a model that takes an MRI scan as input and predicts whether the patient has a brain tumor (and what type).

---

## Technology Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Data handling | torchvision, OpenCV, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model explainability | `pytorch-grad-cam` |
| IDE | Cursor |
| Training platform | Local (with GPU) or Google Colab for heavy training runs |
| Demo UI | Gradio or Streamlit |

---

## Phase 1: Research & Foundation

**Goal:** Understand the problem domain before writing any code.

**What to research:**
- Types of brain tumors visible on MRI (glioma, meningioma, pituitary, no tumor)
- MRI scan types: T1, T2, FLAIR — understand which are most used for tumor detection
- How radiologists read MRI scans
- Medical imaging formats: DICOM and NIfTI

**Where to do it:**
- YouTube: "How to read brain MRI", "brain tumor radiology basics"
- PubMed: search "brain tumor MRI deep learning classification"
- Read any survey paper on "deep learning brain tumor segmentation/classification"

**Deliverable:** 1 page of notes on what the task actually is.

---

## Phase 2: Data Collection

**Goal:** Get labeled MRI scan data.

**Dataset to use:** Kaggle — "Brain Tumor MRI Dataset" by Masoud Nickparvar
- ~7,000 labeled images
- 4 classes: glioma, meningioma, pituitary tumor, no tumor
- Already in PNG format, pre-labeled, split into Training/Testing folders

**Other datasets (if more data is needed later):**

| Dataset | What it has | Where |
|---|---|---|
| BraTS Challenge | High-quality 3D MRI with expert segmentation | synapse.org/brats |
| TCIA | Large medical imaging archive | cancerimagingarchive.net |
| Figshare Brain Tumor | 3,064 T1 MRI images, 3 tumor types | figshare.com |

### Pulling the dataset with the Kaggle API (no manual download needed)

1. Go to kaggle.com → your profile → **Account** → **Create New API Token**
2. Download the `kaggle.json` credentials file
3. Place it at `~/.config/kaggle/kaggle.json` (Mac/Linux) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
4. Run the following in your terminal from the project root:

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
```

5. Then unzip with this script (or run `unzip` in terminal):

```python
import zipfile
with zipfile.ZipFile('brain-tumor-mri-dataset.zip', 'r') as z:
    z.extractall('data/')
```

> **Note:** Add `data/` to your `.gitignore` — do not commit the dataset to GitHub.

### Sanity check — visually inspect a small sample

You don't need to look at all 3,000 images manually, but do look at 20–50 to confirm labels make sense and spot any corrupted files.

```python
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

data_dir = 'data/Training'
classes = os.listdir(data_dir)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax in axes.flatten():
    cls = random.choice(classes)
    img_file = random.choice(os.listdir(f'{data_dir}/{cls}'))
    img = Image.open(f'{data_dir}/{cls}/{img_file}')
    ax.imshow(img, cmap='gray')
    ax.set_title(cls)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**Deliverable:** Dataset downloaded, 20–50 images inspected by hand.

---

## Phase 3: Environment Setup

**Goal:** Get the coding environment ready.

### Workflow: Write in Cursor, Train in Google Colab

Yes — you write and edit all your code locally in Cursor, then upload to Colab to use their free GPU for training. Here's exactly how that works:

1. **Write code in Cursor** — develop your scripts locally (preprocessing, model definition, training loop, evaluation)
2. **Push to GitHub** — commit and push your code from Cursor
3. **Pull into Colab** — at the top of your Colab notebook, clone your repo:
   ```python
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   %cd YOUR_REPO
   ```
4. **Run training in Colab** — Colab provides a free T4 GPU, which is fast enough for this project
5. **Save your trained model** — download the `.pth` checkpoint file or save it to Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   torch.save(model.state_dict(), '/content/drive/MyDrive/brain_tumor_model.pth')
   ```
6. **Pull the model back locally** — download the `.pth` file and use it for the demo UI in Cursor

This is a standard workflow for ML projects. Cursor for code, Colab for compute.

**Local setup (Cursor terminal):**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install torch torchvision numpy matplotlib scikit-learn opencv-python Pillow pytorch-grad-cam gradio
pip freeze > requirements.txt   # save deps so Colab can install them too
```

**In Colab, install from your requirements file:**
```python
!pip install -r requirements.txt
```

**Deliverable:** Local venv working in Cursor, a `requirements.txt` committed to the repo, and a test Colab notebook that clones the repo and imports the dependencies successfully.

---

## Phase 4: Data Preprocessing

**Goal:** Turn raw images into clean, consistent input for the model.

**What to do:**
- Resize all images to 224×224
- Normalize pixel values from 0–255 to 0.0–1.0
- Split into train / validation / test sets (70% / 15% / 15%)
- Apply data augmentation to prevent overfitting: random flips, rotations, brightness shifts
- Handle class imbalance if one class has significantly more examples

**How to do it:**
- `torchvision.transforms` for augmentation and normalization
- `torch.utils.data.DataLoader` for batching

**Deliverable:** A DataLoader that outputs batches of `(image_tensor, label)` pairs.

---

## Phase 5: Model Building

**Goal:** Build the neural network that makes the prediction.

**Approach — Transfer Learning (do not train from scratch):**
- Use a pretrained model (trained on ImageNet): `ResNet50`, `EfficientNetB0`, or `VGG16`
- Replace the final classification layer with one that outputs 4 classes (or 2 for binary)
- Fine-tune the last few layers on the MRI data

**Why transfer learning:** MRI features (edges, textures, shapes) share low-level patterns with natural images. This gets you most of the way there without needing massive compute or data.

**Training config:**
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`, learning rate ~1e-4

**Deliverable:** A model that compiles and runs a forward pass on a dummy input batch.

---

## Phase 6: Training

**Goal:** Train the model and monitor its learning.

**What to do:**
- Train for 10–30 epochs
- Log training loss, validation loss, and validation accuracy each epoch
- Use early stopping (halt if validation loss doesn't improve for 5 consecutive epochs)
- Save the best model checkpoint with `torch.save()`

**Deliverable:** A trained model with >85% validation accuracy, with loss/accuracy plots.

---

## Phase 7: Evaluation

**Goal:** Honestly measure how good the model is.

**Metrics to report:**
- Accuracy (but this alone is misleading for medical data)
- Confusion matrix — see exactly where it fails
- Precision, Recall, F1 — per class
- ROC-AUC — for the binary case
- Grad-CAM — visualize what part of the image the model is looking at

**Why this matters:** A model that predicts "tumor" 100% of the time has high recall but is useless. You must understand failure modes, especially false negatives (missing a real tumor).

**Deliverable:** Evaluation report with confusion matrix and at least 5 Grad-CAM heatmap visualizations.

---

## Phase 8: Demo Interface

**Goal:** Make the model usable without needing to understand the code.

**What to build with Gradio:**
- Input: upload an MRI image
- Output: predicted class + confidence score + Grad-CAM heatmap overlay

```python
import gradio as gr

def predict(image):
    # preprocess → run model → return label + confidence + heatmap
    pass

gr.Interface(fn=predict, inputs="image", outputs=["label", "image"]).launch()
```

**Deliverable:** A shareable demo link anyone can use to test the model.

---

## Suggested Order of Work

```
Phase 1 — Research        2–3 days
Phase 2 — Data            1 day
Phase 3 — Setup           1 day
Phase 4 — Preprocessing   2–3 days
Phase 5 — Model           2–3 days
Phase 6 — Training        1–2 days
Phase 7 — Evaluation      2–3 days
Phase 8 — Demo UI         1 day
```

---

## Where to Start

1. Open Cursor and clone this repo
2. Set up your local virtual environment (see Phase 3)
3. Get your Kaggle API token and download the dataset (see Phase 2)
4. Run the image inspection snippet locally in Cursor to verify the data
5. When you're ready to train (Phase 6), push your code to GitHub and pull it into Google Colab for GPU access
