# 🐾 Vision — AI Animal Classifier

> A production-grade deep learning vision system for multi-class animal classification, built collaboratively using **Transfer Learning** on the **Animals-10** dataset.

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-API-D00000?logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/MobileNetV2-ImageNet-34A853?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" />
</p>

---

## 👥 Team & Roles

| Role | Name |
|---|---|
| **Project Lead / ML Architecture** | Jaydeep Ahir |
| **Data Engineering / Pipeline** | Jihan Gajjar |

---

## 📌 Project Objective

This project is a **collaborative AI vision classifier** designed to accurately identify animal species from photographic images. Trained on the **Animals-10** dataset — comprising **30,000+ high-resolution images** across 10 distinct classes — the system leverages **Transfer Learning** with Google's **`MobileNetV2`** architecture pre-trained on ImageNet to achieve high classification accuracy with efficient computational overhead.

The 10 target classes are:

`🐕 Dog` · `🐎 Horse` · `🐘 Elephant` · `🦋 Butterfly` · `🐔 Chicken` · `🐈 Cat` · `🐄 Cow` · `🐑 Sheep` · `🕷️ Spider` · `🐿️ Squirrel`

> **Note on AI Assistance:** The ML architecture, data engineering pipeline, and core code execution are original collaborative work. Explanatory markdown documentation was drafted with the assistance of an AI language model for clarity and professionalism.

---

## 🏗️ System Architecture — The Narrative

The project is structured as a **two-phase collaborative pipeline**, where each phase represents a distinct engineering discipline. The final **Master Notebook** (`Vision_Master.ipynb`) merges both phases into a single, end-to-end executable workflow.

```
📂 Repository Structure
├── Data_Pipeline_Engineering.ipynb    # Phase 1 — Jihan's data engineering work
├── Model_Architecture_and_Training.ipynb  # Phase 2 — Jaydeep's ML architecture
├── Vision_Master.ipynb                # Merged master notebook (end-to-end)
└── README.md                          # This file
```

---

### 🔹 Phase 1: Data Engineering & Preprocessing Pipeline
> *Engineered by Jihan Gajjar*

The foundation of any robust ML system is its data pipeline. Phase 1 establishes a **production-ready data ingestion and augmentation system** built entirely on Keras's **`ImageDataGenerator`** API.

**Key Engineering Decisions:**

- **📥 Dataset Acquisition:** The **Animals-10** dataset is programmatically downloaded via [`kagglehub`](https://github.com/Kaggle/kagglehub), ensuring reproducibility across environments. The raw dataset contains Italian-language folder names, which are mapped to English labels through a **global class dictionary** — adhering to the DRY (Don't Repeat Yourself) principle.

- **🔄 Real-Time Data Augmentation:** To combat overfitting on the training set, the pipeline applies **on-the-fly stochastic transformations** to training images at every epoch, effectively generating an infinite stream of unique training samples:
  - **`rotation_range=20`** — Random rotations up to ±20°
  - **`zoom_range=0.2`** — Random zoom transformations up to 20%
  - **`horizontal_flip=True`** — Random horizontal mirroring

- **🔒 Validation Integrity:** The validation generator is configured with **rescaling only** (`rescale=1./255`) — no augmentation, no shuffling (`shuffle=False`). This ensures that evaluation metrics are computed on clean, deterministic data, and that the **Confusion Matrix** in Phase 3 produces accurate per-class analysis.

- **📊 Pipeline Verification:** A visual batch inspection step renders a grid of augmented training images with their English labels, confirming the pipeline is feeding correctly formatted data before any model training begins.

**Technical Stack:**
| Component | Implementation |
|---|---|
| Image Size | `224 × 224 × 3` (RGB) |
| Batch Size | `32` |
| Train/Val Split | `80% / 20%` (via `validation_split=0.2`) |
| Pixel Normalization | `rescale=1./255` |
| Class Mode | `categorical` (one-hot encoded) |

---

### 🔹 Phase 2: ML Architecture & Training Protocol
> *Engineered by Jaydeep Ahir*

Phase 2 consumes the augmented data pipeline from Phase 1 and constructs a high-accuracy classifier through a **two-step training protocol** — a proven strategy in transfer learning literature.

**🧠 Neural Network Architecture:**

```
┌─────────────────────────────────────────┐
│         MobileNetV2 (ImageNet)          │  ← Pre-trained feature extractor
│     Input: (224, 224, 3) — RGB          │     1,280 convolutional filters
│     include_top=False                   │
├─────────────────────────────────────────┤
│       GlobalAveragePooling2D()          │  ← Spatial dimensionality reduction
├─────────────────────────────────────────┤
│         Dense(256, relu)                │  ← Custom classification head
├─────────────────────────────────────────┤
│           Dropout(0.4)                  │  ← Regularization (40% drop rate)
├─────────────────────────────────────────┤
│      Dense(10, softmax)                 │  ← Output: 10-class probability
└─────────────────────────────────────────┘
```

**🔬 Two-Step Training Protocol:**

| Step | Strategy | Epochs | Learning Rate | Description |
|---|---|---|---|---|
| **Phase 1 — Feature Extraction** | Base model **frozen** | `10` | `1e-3` | Only the custom dense head is trained. The pre-trained MobileNetV2 weights act as a fixed feature extractor, allowing the classifier to learn task-specific decision boundaries quickly. |
| **Phase 2 — Fine-Tuning** | Top **50 layers unfrozen** | `5` | `1e-5` | The upper convolutional layers of MobileNetV2 are unfrozen and retrained with a **micro-learning rate** (`0.00001`). This delicately adapts the pre-trained ImageNet representations to the specific visual features of the Animals-10 dataset without catastrophic forgetting. |

**Why this approach?**
- Training the head first ensures the randomly initialized dense layers converge to reasonable weights *before* the base model weights are modified.
- The 100× reduction in learning rate during fine-tuning (`1e-3` → `1e-5`) prevents large gradient updates that could destroy the valuable pre-trained feature representations.

**📈 Training Visualization:** Accuracy and loss curves for both training and validation sets are plotted after each phase to monitor convergence and detect overfitting early.

---

## 📦 Pre-Trained Models (Hugging Face)

> [!IMPORTANT]
> Due to **GitHub's 100MB file size limit**, the production-ready model weights are hosted externally on Hugging Face Hub. The trained model exceeds this limit and cannot be stored directly in this repository.

Both the **classic** and **modern** Keras serialization formats are provided:

| Format | File | Description |
|---|---|---|
| **HDF5** (`.h5`) | `Zenity_Vision_Classifier.h5` | Classic Keras format. Broad compatibility with TensorFlow 2.x and legacy tooling. |
| **Native Keras** (`.keras`) | `Zenity_Vision_Classifier.keras` | Modern Keras 3 format. Recommended for new projects — supports cross-framework portability. |

### ⬇️ Download

➡️ **[Download Models from Hugging Face](https://huggingface.co/jaydeepahir18/Vision-Animal-Classifier/tree/main)**

**Quick Load Example:**
```python
from tensorflow.keras.models import load_model

# Option 1: HDF5 format
model = load_model("Zenity_Vision_Classifier.h5")

# Option 2: Native Keras format (recommended)
model = load_model("Zenity_Vision_Classifier.keras")
```

---

## 📊 Evaluation & Testing

### Confusion Matrix
The master notebook generates a **full 10×10 Confusion Matrix** using `sklearn.metrics.confusion_matrix`, visualized as a heatmap via **Seaborn**. This provides granular insight into:
- Per-class precision and recall
- Specific inter-class misclassification patterns (e.g., distinguishing cats from dogs)
- Overall model robustness across all 10 animal categories

A detailed **Classification Report** (precision, recall, F1-score per class) is also printed for quantitative analysis.

### 🖼️ Interactive Inference — Test with Your Own Images
The final cell of `Vision_Master.ipynb` includes a **fully interactive image upload widget** powered by `google.colab.files.upload()`. Users can:

1. Click the **"Choose Files"** button to upload one or more images
2. The model preprocesses each image to `224×224` and normalizes pixel values
3. Predictions are displayed with the **English class label** and **confidence percentage**
4. Results are color-coded: <span style="color:green">**🟢 Green**</span> for confidence > 80%, <span style="color:darkorange">**🟠 Orange**</span> for lower confidence

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open **`Vision_Master.ipynb`** in [Google Colab](https://colab.research.google.com/)
2. Ensure the runtime is set to **GPU** for accelerated training:
   - `Runtime` → `Change runtime type` → `T4 GPU`
3. Click **`Runtime` → `Run all`** to execute the full pipeline end-to-end
4. When the final cell executes, use the **upload widget** to test the model with your own animal images

### Option 2: Run Individual Notebooks
For a closer look at each contributor's work:
- **`Data_Pipeline_Engineering.ipynb`** — Run to inspect the data augmentation pipeline independently
- **`Model_Architecture_and_Training.ipynb`** — Run to focus on the neural network architecture and training protocol

### Prerequisites
```bash
pip install tensorflow kagglehub matplotlib numpy scikit-learn seaborn
```

> [!NOTE]
> The dataset (~700MB) is automatically downloaded from Kaggle via `kagglehub` at runtime. A Kaggle API key may be required on first use. In Google Colab, this is typically handled via an interactive prompt.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning Framework** | TensorFlow / Keras |
| **Pre-trained Model** | MobileNetV2 (ImageNet) |
| **Data Augmentation** | `ImageDataGenerator` |
| **Evaluation** | scikit-learn, Seaborn, Matplotlib |
| **Dataset** | [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) (30,000+ images) |
| **Platform** | Google Colab (GPU-accelerated) |
| **Model Hosting** | Hugging Face Hub |
| **Language** | Python 3.10+ |

---

## 📜 License

This project is intended for **academic and educational purposes**.

---

<p align="center">
  <b>Built by Jaydeep Ahir & Jihan Gajjar</b>
</p>
