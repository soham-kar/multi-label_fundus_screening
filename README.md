# GLAAM-4X: Disease-Specific Multi-Scale Attention for Multi-Label Fundus Disease Detection

> **GLAAM-4X** is a multi-label deep learning classifier that detects four sight-threatening ocular diseases — **Cataract**, **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Myopia** — from a single colour fundus photograph. It extends the [GLAAM attention module](https://doi.org/10.1016/j.cmpbup.2025.100182) with disease-specific attention specialists, a multi-scale DR head, and Asymmetric Loss for extreme class imbalance.

---

## Results

| Disease | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Cataract | **0.985** | 0.873 | 0.862 | 0.885 |
| DR | **0.952** | 0.848 | 0.860 | 0.837 |
| Glaucoma | **0.966** | 0.690 | 0.833 | 0.588 |
| Myopia | **0.971** | 0.732 | 0.750 | 0.714 |
| **Macro avg** | **0.968** | **0.786** | 0.826 | 0.756 |

- **Model:** 12.3M parameters · 47 MB · 12.44 ms inference latency (Tesla T4 GPU)
- **Training data:** 27,899 images from 9 public datasets (ODIR-5K, IDRiD, DDR, RFMiD, JSIEC, PAPILA, REFUGE2, glaucoma_bundle, eye_diseases)
- **Test set:** 2,232 held-out images

---

## Repository Structure

```
├── demo/                        # Flask web app for interactive inference
│   ├── app.py                   # Flask server (upload image → bar plot + heatmap)
│   ├── templates/index.html     # Light-themed UI with drag-drop upload
│   ├── model_weights.pth        # Trained GLAAM-4X weights (47 MB)
│   ├── thresholds.json          # Per-disease optimal thresholds
│   ├── models/glaam_4x.py       # Model architecture definition
│   ├── demo_images/             # 5 sample fundus images (one per disease + normal)
│   ├── predict.py               # Standalone CLI inference script
│   └── requirements.txt         # Minimal dependencies for demo
│
├── notebooks/                   # Training notebooks (Google Colab)
│   ├── modal_train_glaam4x_v4_kaggle.ipynb   # Full training pipeline (60 epochs)
│   └── train_on_colab.ipynb                  # Earlier training variant
│
├── results/                     # Training outputs and evaluation artifacts
│   ├── training_graphs/         # 11 publication-quality figures (300 DPI)
│   ├── report_assets/           # 7 test-set figures + results table CSV
│   ├── config.json              # Training hyperparameters
│   ├── results.json             # Final test-set metrics
│   └── model_efficiency_summary.json
│
└── README.md                    # This file
```

---

## Quick Start: Run the Demo

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (CPU or CUDA)
- 550 MB disk space (for dependencies)

### Installation

```bash
cd demo
pip install -r requirements.txt
```

### Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Using the Demo

1. **Upload a fundus image** — drag and drop or click "Choose Image"
2. **Or try a sample** — click one of the 5 sample buttons (Cataract, DR, Glaucoma, Myopia, Normal)
3. **Click "Analyze Image"**
4. View the results:
   - **Bar plot** — side-by-side original image + horizontal bar chart of 4 disease probabilities with threshold markers
   - **Disease cards** — per-disease probability bars with DETECTED badges
   - **EigenGradCAM heatmap** — 3-panel visualisation (original, heatmap, overlay) showing where the model looked
   - **Meta info** — inference time, device, model details

### CLI Inference

```bash
cd demo
python predict.py --weights model_weights.pth --image fundus.jpg --thresholds thresholds.json --device cuda
```

### Python API

```python
import sys
sys.path.insert(0, 'demo')

from predict import load_model, predict

model = load_model("demo/model_weights.pth", device="cuda")
results = predict(model, "fundus.jpg", thresholds_path="demo/thresholds.json")
print(results)
# {"Cataract": {"probability": 0.985, "prediction": 1}, ...}
```

---

## Model Architecture

GLAAM-4X extends the original GLAAM [7] with four key innovations:

### 1. Disease-Specific Attention Specialists

Each disease receives a dedicated attention pathway on a shared MobileNetV2 backbone:

| Disease | Attention type | Reduction ratio | Rationale |
|---|---|---|---|
| DR | MultiScaleGLAAM (3 scales) | 4 / 8 / 16 | Lesions span 2–50+ px (microaneurysms to haemorrhages) |
| Glaucoma | GLAAMBlock | 8 | Optic disc is a localised, medium-scale structure |
| Cataract | GLAAMBlock | 16 | Lens opacity is diffuse and global |
| Myopia | Identity (no attention) | — | Tessellation already captured by backbone |

### 2. MultiScaleGLAAM for DR

Three-scale attention captures the full range of DR lesion sizes:
- **Fine** (1× resolution, r=4): microaneurysms (2–5 px)
- **Medium** (2× downsampled, r=8): haemorrhages (5–20 px)
- **Coarse** (4× downsampled, r=16): large lesions (20–50+ px)

### 3. Post-Backbone Integration

Attention is applied *after* the MobileNetV2 backbone (not intra-backbone), preserving pretrained features and enabling efficient multi-disease branching from a single shared 1280-channel feature map. The four specialists add only 2.3M parameters (18% of total).

### 4. Asymmetric Loss

| Parameter | Value | Purpose |
|---|---|---|
| γ_neg | 4.0 | Aggressively suppress easy negative gradients |
| γ_pos | 0.0 | Preserve full gradient from all positive samples |
| Clip m | 0.05 | Prevent overconfidence on dominant negative class |

---

## Training

### Training Notebook

The full training pipeline is in `notebooks/modal_train_glaam4x_v4_kaggle.ipynb`. It is designed for Google Colab with a Tesla T4 GPU.

### Training Configuration

| Parameter | Value |
|---|---|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Input size | 384 × 384 |
| Batch size | 32 |
| Optimiser | AdamW (lr = 2e-5, weight decay = 5e-4) |
| LR schedule | 10-epoch warmup → cosine annealing (60 epochs) |
| Loss | Asymmetric Loss (γ_neg=4.0, γ_pos=0.0, clip=0.05) |
| Augmentation | Differential (stronger for disease-positive images) |
| Sampling | WeightedRandomSampler (√-inverse frequency) |
| Early stopping | Patience = 10 |
| Best epoch | 48 |
| Mixed precision | AMP (float16) |

### Datasets

Nine public Kaggle datasets were unified for training:

| Dataset | Diseases | Images |
|---|---|---|
| ODIR-5K | Cataract, DR, Glaucoma, Myopia | ~5,000 |
| IDRiD | DR | ~500 |
| DDR | DR | ~1,300 |
| RFMiD | DR, Glaucoma, Myopia | ~3,200 |
| JSIEC | DR, Glaucoma, Myopia | ~1,000 |
| PAPILA | Glaucoma | ~400 |
| REFUGE2 | Glaucoma | ~200 |
| glaucoma_bundle | Glaucoma | ~1,100 |
| eye_diseases | Cataract, DR, Glaucoma | ~4,000 |

**Total after merge + dedup:** 27,899 images (80/10/10 split)

---

## Results Folder

The `results/` folder contains all training and evaluation artifacts:

### Training Graphs (`training_graphs/`)

| File | Content |
|---|---|
| `01_loss_curve.png` | Training and validation loss |
| `02_macro_f1_progression.png` | Macro F1 (train, val, val-optimal) |
| `03_per_disease_auc.png` | Per-disease validation AUC |
| `04_per_disease_f1.png` | Per-disease validation F1 |
| `05_precision_recall.png` | Per-disease precision vs recall |
| `06_lr_schedule.png` | Learning rate schedule |
| `07_dashboard.png` | Combined 6-panel dashboard |
| `08_train_val_loss.png` | Overfitting detection |
| `09_roc_curves.png` | ROC curves (validation) |
| `10_pr_curves.png` | PR curves (validation) |
| `11_confusion_matrices.png` | Confusion matrices (validation) |

### Report Assets (`report_assets/`)

| File | Content |
|---|---|
| `01_dataset_composition.png` | Split sizes + per-disease prevalence |
| `02_test_roc_curves.png` | ROC curves (test set) |
| `03_test_pr_curves.png` | PR curves (test set) |
| `04_test_confusion_matrices.png` | Confusion matrices (test set) |
| `05_calibration_curves.png` | Calibration curves |
| `06_results_table.png` | Master results table |
| `results_table.csv` | Results as CSV |
| `model_efficiency_summary.json` | Parameters, size, latency |

---

## Explainability

GLAAM-4X generates **disease-specific EigenGradCAM heatmaps** showing where the model focused for each detected disease:

- **Cataract:** Diffuse attention across the entire fundus (global lens opacity)
- **Glaucoma:** Concentrated at the optic disc region
- **DR:** Localised to lesion-bearing retinal regions
- **Myopia:** Low, diffuse attention (no specific lesion)

The demo app generates these heatmaps automatically for each prediction.

---

## Citation

If you use this work, please cite:

```bibtex
@article{glaam4x2026,
  title={GLAAM-4X: Disease-Specific Multi-Scale Attention for Multi-Label Fundus Disease Detection from a Single Retinal Image},
  year={2026}
}
```

And the original GLAAM paper:

```bibtex
@article{kumar2025glaam,
  title={GLAAM and GLAAI: Pioneering attention models for robust automated cataract detection},
  author={Kumar, Deepak and Verma, Chaman and Illés, Zoltán},
  journal={Computer Methods and Programs in Biomedicine Update},
  volume={7},
  pages={100182},
  year={2025},
  doi={10.1016/j.cmpbup.2025.100182}
}
```

---

## License

MIT License

---

## Medical Disclaimer

This tool is for research and demonstration purposes only and does not constitute medical advice. Always consult a qualified ophthalmologist for diagnosis and treatment.
