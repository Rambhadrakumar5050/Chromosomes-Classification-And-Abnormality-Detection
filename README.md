
# 🧬 Chromosome Classification & Abnormality Detection

A deep learning system for automated classification of human chromosomes into 24 classes (chromosomes 1–22, X, Y) and per-chromosome abnormality detection using transfer learning, multi-model comparison, and ensemble learning.

Built as part of a Summer Internship project in medical image analysis.

---
 
## Project Overview

Human karyotype analysis involves identifying and classifying all 24 types of chromosomes from microscope images, and detecting structural or numerical abnormalities (e.g. trisomy, deletion, translocation). This project automates that process using convolutional neural networks trained on a labeled chromosome image dataset.

**Key goals:**
- Classify each chromosome into one of 24 classes with >90% accuracy
- Flag chromosomes as normal or potentially abnormal per image
- Compare 5 model architectures and select the best
- Achieve >90% inference accuracy using ensemble learning
- Show results with graphs.
---

## Dataset

- **Source:** Labeled chromosome image dataset (single chromosomes, cropped and annotated) and BioIMLAB dataset cinsisting of single chromosome images.
- **Structure:**
  - `single_chromosomes_object/JEPG/` — individual chromosome images
  - `single_chromosomes_object/anntations/` — XML annotations
  - `normal.csv` — patient IDs with normal karyotypes
  - `number_abnormalities.csv` — patients with numerical abnormalities (e.g. trisomy 21)
  - `structural_abnormalities.csv` — patients with structural abnormalities
- **Classes:** 24 (chromosomes 1–22, X, Y), grouped by Denver classification (A–G + Sex)
- **Label parsing:** Chromosome class is encoded in the filename (`digits[-3:-1]` = 2-digit chromosome number)
- **Split:** 70% train / 15% validation / 15% test (stratified)

---

## Models Trained

Five architectures were trained and compared using the same classification head and training strategy:

| Model | Backbone | Parameters | Notes |
|---|---|---|---|
| EfficientNetB3 | EfficientNet | ~12M | Best accuracy/efficiency tradeoff |
| ResNet50 | ResNet | ~25M | Strong skip connections, stable training |
| VGG16 | VGG | ~138M | High capacity, slower inference |
| MobileNetV2 | MobileNet | ~3.4M | Fastest inference, lightest model |
| **Ensemble** | All 4 combined | — | Soft voting across all models |

All models use ImageNet pretrained weights and follow a two-phase training strategy.

---

## Training Strategy

### Phase 1 — Frozen Backbone (Feature Extraction)
- Backbone weights frozen, only the classification head trains
- Learning rate: `1e-3`
- Epochs: 20
- Purpose: Learn chromosome-specific features quickly without disturbing ImageNet weights

### Phase 2 — Fine-Tuning (Top Layers Unfrozen)
- Top layers of the backbone unfrozen (model-specific cutoff layers)
- Learning rate: `1e-4` (10x lower to avoid catastrophic forgetting)
- Epochs: 60 to 80
- BatchNorm layers in backbone kept frozen (best practice for small-medium datasets)
- Early stopping restores best weights automatically

---

## Optimization Techniques

| Technique | Details |
|---|---|
| Weight initialization | He Uniform for Dense layers, Glorot Uniform for output layer |
| Batch Normalization | After every Dense layer, before activation — prevents vanishing gradients |
| Dropout | 0.40 (fc1), 0.30 (fc2) — prevents overfitting |
| L2 Regularization | `1e-4` on all Dense weights |
| Optimizer | Adam (β1=0.9, β2=0.999, ε=1e-7) |
| Label Smoothing | 0.05 — prevents overconfident predictions |
| LR Scheduling | ReduceLROnPlateau (factor=0.3, patience=4) |
| Early Stopping | patience=8 (Phase 1), 12 (Phase 2), restores best weights |
| Class Weights | Computed from training distribution to handle class imbalance |
| Data Augmentation | Random flip, rotation (±90°), zoom, brightness/contrast jitter, translation |
| Mixed Precision | float16 on GPU (float32 on CPU) for faster training |

### Vanishing Gradient Prevention
- He Uniform initialization sets variance correctly for ReLU activations
- Batch Normalization normalizes layer inputs, stabilizing gradient flow
- ResNet50 and EfficientNetB3 backbones use skip connections architecturally
- Backbone BatchNorm frozen during fine-tuning to avoid instability

---

## Ensemble Learning

Three ensemble strategies were evaluated:

1. **Soft Voting (Average)** — averages the softmax probability vectors across all 4 models
2. **Weighted Voting** — weights each model by its test accuracy before averaging
3. **Hard Majority Voting** — takes the most common predicted class across models

The soft voting ensemble consistently outperformed all individual models and was used for final inference.

---

## Abnormality Detection

Each chromosome is flagged as normal or potentially abnormal based on:

- **Low confidence** — softmax confidence below threshold (0.70) → flagged as "Low Confidence — Possibly Abnormal"
- **High entropy** — prediction entropy above 50% of maximum entropy → "High Entropy — Ambiguous Morphology"
- **CSV ground truth** — patient-level labels from `normal.csv`, `number_abnormalities.csv`, `structural_abnormalities.csv`

---

## Results

| Model | Test Accuracy | Top-3 Accuracy | Macro F1 | Inference (ms/img) |
|---|---|---|---|---|
| EfficientNetB3 | ~% | ~% | ~.## | ~ms |
| ResNet50 | ~% | ~% | ~.## | ~ms |
| VGG16 | ~% | ~% | ~.## | ~ms |
| MobileNetV2 | ~% | ~% | ~.## | ~ms |
| Ensemble (Soft) | ~% | ~% | ~.## | N/A |

> Results will be updated after full-scale training completes.

---

## Visualizations Produced

- Class distribution bar chart
- Training curves (loss and accuracy) per model per phase
- Confusion matrices (count and percentage) for all models
- Per-class precision / recall / F1 bar charts
- Sample prediction grid (green = correct + normal, orange = correct + abnormal, red = wrong)
- GradCAM heatmaps showing which regions the model focuses on

---

## Project Structure
chromosome-classification/
├── Chromosome_Demo.ipynb               # Small-scale proof of concept (15 images/class, CPU)
├── Chromosome_FullScale_Production.ipynb  # Full training: all 5 models + ensemble
├── data/
│   ├── single_chromosomes_object/
│   │   ├── JEPG/                       # Chromosome images
│   │   └── anntations/                 # XML bounding box annotations
│   ├── normal.csv
│   ├── number_abnormalities.csv
│   └── structural_abnormalities.csv
└── outputs/
├── *_final.keras                   # Saved model files
├── model_comparison.csv            # Accuracy table across all models
├── ensemble_predictions.csv        # Per-image inference results
├── *_confusion.png                 # Confusion matrices
├── *_curves.png                    # Training curves
├── *_per_class.png                 # Per-class metrics
├── *_gradcam.png                   # GradCAM explanations
└── sample_predictions.png          # Prediction visualization grid

---

## How to Run

### Google Colab (Recommended — free GPU)

1. Upload `Chromosome_FullScale_Production.ipynb` to Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Mount Google Drive and update `DATA_ROOT` in Cell 3
4. Run all cells top to bottom

### Local (CPU only)

```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas tqdm
```

Update `DATA_ROOT` in Cell 3 to your local dataset path, reduce `BATCH_SIZE` to 8–16.

---

## Requirements
-tensorflow >= 2.12
-opencv-python
-scikit-learn
-matplotlib
-seaborn
-pandas
-numpy
-tqdm

---

## Key Concepts

**Denver Classification** groups chromosomes by size and centromere position:
- Group A: 1, 2, 3 | Group B: 4, 5 | Group C: 6–12 | Group D: 13, 14, 15
- Group E: 16, 17, 18 | Group F: 19, 20 | Group G: 21, 22 | Sex: X, Y

**CDA (Chromosome Data Augmentation):** Chromosomes are orientation-invariant (a chromosome flipped or rotated 180° is the same chromosome), so aggressive rotation and flip augmentation is biologically valid and significantly improves generalization.

**GradCAM (Gradient-weighted Class Activation Mapping):** Highlights which pixels of the chromosome image contributed most to the model's prediction — useful for clinical explainability.

---

## Internship Context

This project was developed during a Summer Internship focused on medical image analysis. The demo notebook (`Chromosome_Demo.ipynb`) was built first on a small data subset (15 images/class, 64×64, CPU) to validate the pipeline end-to-end. The production notebook (`Chromosome_FullScale_Production.ipynb`) scales this up to the full dataset with five model architectures, all optimization techniques, and ensemble learning.

---

## Acknowledgements

- Dataset provided as part of the internship program
- Pretrained weights from ImageNet via `tf.keras.applications`
- GradCAM implementation adapted from the Keras documentation
