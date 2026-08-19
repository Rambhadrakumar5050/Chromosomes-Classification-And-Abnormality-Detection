# 🧬 Chromosome Classification Using Deep Learning

A deep learning-based medical image classification system for the **automated classification of individual human chromosome images into 24 chromosome classes (1–22, X, Y)**.

The project focuses on developing a robust image-processing and deep-learning pipeline capable of learning morphological characteristics of chromosomes from microscopic images.

---
 
## 📌 Project Overview

Chromosome classification is an important step in cytogenetic and karyotype analysis. Traditionally, chromosome identification is performed manually by experts based on characteristics such as chromosome size, shape, centromere position, and banding patterns.

This project aims to automate the classification of **single chromosome images** using image preprocessing and deep learning.

The current pipeline consists of:

```text
Single Chromosome Image
        ↓
       Crop
        ↓
      CLAHE
        ↓
Gaussian Filtering
        ↓
    Sharpening
        ↓
 Data Augmentation
        ↓
 Deep Learning Model
        ↓
 24-Class Prediction
```

### Key Objectives

* Classify individual chromosomes into **24 classes**
* Classes include chromosomes **1–22, X and Y**
* Improve image quality using sequential preprocessing
* Reduce overfitting using data augmentation and regularization
* Train and compare multiple CNN architectures
* Evaluate models using accuracy, loss, confusion matrix and per-class metrics
* Develop a robust chromosome classification pipeline for medical image analysis

---

# 📊 Dataset

The dataset consists of individual chromosome microscope images.

### Dataset Statistics

| Property           |         Value |
| ------------------ | ------------: |
| Total Images       |     **5,474** |
| Number of Classes  |        **24** |
| Image Size         | **224 × 224** |
| Image Type         |     Grayscale |
| Pixel Range        |     **0–255** |
| Minimum Class Size |        **45** |
| Maximum Class Size |       **238** |

### Classes

```text
Class_1
Class_2
Class_3
Class_4
Class_5
Class_6
Class_7
Class_8
Class_9
Class_10
Class_11
Class_12
Class_13
Class_14
Class_15
Class_16
Class_17
Class_18
Class_19
Class_20
Class_21
Class_22
Class_X
Class_Y
```
<img width="931" height="653" alt="image" src="https://github.com/user-attachments/assets/ee2ddb21-8284-4da3-a457-2dd482ac492d" />

These correspond to the 22 autosomal chromosome types and the two sex chromosomes.

---

# 📈 Class Distribution

The dataset is not perfectly balanced.

### Main classes

* Classes 1–22: approximately **238 images per class**
* Class X: **193 images**
* Class Y: **45 images**

| Class Group  | Number of Images |
| ------------ | ---------------: |
| Classes 1–22 |        ~238 each |
| Class X      |              193 |
| Class Y      |               45 |

The significantly smaller number of **Class Y** images creates a class-imbalance problem that is considered during model training.

---

# ✂️ Data Splitting

The complete dataset is divided into training, validation and testing subsets.

| Dataset    |    Images |
| ---------- | --------: |
| Training   | **3,831** |
| Validation |   **821** |
| Testing    |   **822** |
| Total      | **5,474** |

Approximate split:

```text
70% Training
15% Validation
15% Testing
```

The split is performed while maintaining class representation as much as possible.

---

# 🖼️ Image Preprocessing

Because the original chromosome images are relatively low-contrast and blurry, a sequential preprocessing pipeline is used.

## Preprocessing Pipeline

```text
Original Image
      ↓
     Crop
      ↓
     CLAHE
      ↓
Gaussian Filtering
      ↓
  Sharpening
      ↓
Preprocessed Image
```

### 1. Cropping

Cropping is used to focus the image on the chromosome region and remove unnecessary background.

This helps the model concentrate on chromosome morphology rather than large areas of black background.

---

### 2. CLAHE

**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is used to improve local contrast.

It enhances subtle chromosome structures while limiting excessive contrast amplification.

Benefits:

* Improves local contrast
* Makes chromosome regions more distinguishable
* Enhances structural details
* Helps compensate for uneven illumination

---

### 3. Gaussian Filtering

Gaussian filtering is applied after CLAHE to reduce high-frequency noise.

Benefits:

* Reduces image noise
* Smooths unwanted artifacts
* Produces cleaner chromosome structures
* Helps stabilize subsequent sharpening

---

### 4. Sharpening

A sharpening operation is applied after Gaussian filtering to enhance chromosome boundaries and structural details.

The combination:

```text
CLAHE → Gaussian → Sharpening
```

provides a balance between contrast enhancement, noise reduction and edge enhancement.

---

# 🔬 Preprocessing Visualization

Different chromosome samples are visualized before and after preprocessing to verify the effect of the pipeline.

Example comparison:

```text
Original
   ↓
CLAHE
   ↓
Gaussian
   ↓
Sharpening
   ↓
Final Preprocessed Image
```

Multiple chromosome classes are used during visualization rather than relying on a single image, allowing the preprocessing pipeline to be evaluated across different chromosome shapes and sizes.

---

# 🔄 Data Augmentation

The chromosome dataset is relatively small and contains significant class imbalance, particularly for **Class Y**.

Therefore, data augmentation is being incorporated to increase training diversity and reduce overfitting.

Planned/implemented augmentation techniques include:

* Random rotation
* Horizontal flipping
* Vertical flipping where appropriate
* Random zoom
* Translation
* Brightness variation
* Contrast variation

Chromosomes are approximately orientation-invariant for classification purposes, making rotation and flipping particularly useful augmentation strategies.

### Important

Data augmentation is applied to the **training data only**.

Validation and test images are kept unchanged apart from the required preprocessing and normalization.

---

# 🧠 Deep Learning Models

Multiple CNN architectures are being considered for chromosome classification.

## Current Baseline

### ResNet18

The first baseline experiment uses **ResNet18**.

ResNet18 was selected because it provides:

* Residual/skip connections
* Relatively low computational cost
* Fast training
* Strong feature extraction capabilities
* A useful baseline for comparison

---

## Additional Models

The project will also evaluate deeper and more efficient architectures:

### ResNet50

A deeper residual network capable of learning more complex morphological features.

### InceptionV3

Uses multi-scale convolutional feature extraction through Inception modules.

This can help capture chromosome structures appearing at different spatial scales.

### EfficientNet

EfficientNet architectures provide an effective balance between:

* Model depth
* Width
* Input resolution
* Number of parameters
* Computational efficiency

The current model-comparison stage will evaluate these architectures under a consistent training pipeline.

---

# ⚙️ Model Training Strategy

The model is trained as a multi-class classification problem.

```text
Input
224 × 224 × 1
      ↓
CNN Backbone
      ↓
Feature Extraction
      ↓
Fully Connected Layer
      ↓
24-Class Output
      ↓
Softmax
```

The final layer contains **24 output neurons**, corresponding to:

```text
1, 2, 3, ..., 22, X, Y
```

---

# 📉 Initial ResNet18 Baseline Results

The first ResNet18 experiment revealed significant overfitting.

### Training Results

| Epoch | Train Accuracy | Validation Accuracy |
| ----: | -------------: | ------------------: |
|     1 |        100.00% |              81.36% |
|     2 |        100.00% |              80.39% |
|     3 |        100.00% |          **82.10%** |
|     4 |        100.00% |              81.97% |
|     5 |        100.00% |              81.36% |
|     6 |        100.00% |              81.36% |
|     7 |        100.00% |              81.00% |
|     8 |        100.00% |              80.76% |
|     9 |        100.00% |              80.76% |
|    10 |        100.00% |              81.49% |

### Best Validation Accuracy

```text
82.10%
```

The model reaches almost perfect training accuracy very quickly while validation accuracy remains around 80–82%.

This indicates **strong overfitting**.

---

# ⚠️ Current Challenge: Overfitting

The major challenge currently observed is the gap between training and validation performance.

```text
Training Accuracy      ≈ 100%
Validation Accuracy    ≈ 82%
```

Possible contributing factors include:

* Limited dataset size
* Significant class imbalance
* Very similar chromosome structures
* Blurry/low-resolution source images
* High model capacity relative to dataset size
* Insufficient training diversity
* Difficulty distinguishing visually similar chromosome classes

The original images themselves contain limited visual information, meaning preprocessing alone cannot completely recover details that were not captured in the original image.

---

# 🛠️ Overfitting Reduction Strategy

The next training stage focuses on improving generalization rather than simply increasing the number of epochs.

## Techniques Being Applied

### 1. Stronger Data Augmentation

Increase the diversity of chromosome appearances seen during training.

### 2. Dropout

Dropout will be added to the classification head to reduce dependence on specific neurons.

### 3. Weight Decay / L2 Regularization

Regularization will penalize excessively large weights and encourage simpler models.

### 4. Label Smoothing

Label smoothing will reduce excessive confidence in predictions and improve generalization.

### 5. Early Stopping

Training will stop when validation performance stops improving instead of continuing until the model memorizes the training set.

### 6. Class Weighting

Class weights will be used to give more importance to underrepresented classes.

This is especially important for:

```text
Class X → 193 images
Class Y → 45 images
```

### 7. Oversampling

Oversampling of minority classes may also be evaluated to provide the model with more examples of underrepresented chromosome types.

### 8. Transfer Learning / Layer Freezing

For pretrained architectures, the backbone will initially be frozen and selected layers will later be unfrozen for fine-tuning.

This allows the model to learn chromosome-specific features without immediately destroying useful pretrained features.

---

# 🧪 Model Comparison

The final pipeline will compare multiple architectures under similar preprocessing and training conditions.

| Model        | Purpose                        |
| ------------ | ------------------------------ |
| ResNet18     | Initial baseline               |
| ResNet50     | Deeper residual architecture   |
| InceptionV3  | Multi-scale feature extraction |
| EfficientNet | Efficient feature extraction   |

The best model will be selected based on validation and test performance rather than training accuracy alone.

---

# 📊 Evaluation Metrics

The models will be evaluated using multiple metrics.

### Accuracy

Overall percentage of correctly classified chromosome images.

### Precision

Measures how many predicted samples of a class actually belong to that class.

### Recall

Measures how many samples belonging to a class are correctly identified.

### F1 Score

Harmonic mean of precision and recall.

### Confusion Matrix

Used to identify which chromosome classes are frequently confused with each other.

### Per-Class Accuracy

Used to identify weak performance on individual chromosome classes, especially minority classes.

---

# 📈 Planned Visualizations

The project generates visualizations to understand model behavior and performance.

### Dataset Analysis

* Class distribution
* Sample chromosome images
* Preprocessing comparisons

### Training Analysis

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

### Classification Analysis

* Confusion matrix
* Normalized confusion matrix
* Per-class precision
* Per-class recall
* Per-class F1 score
* Classification report

### Model Comparison

* Validation accuracy comparison
* Test accuracy comparison
* Parameter count
* Training time
* Inference time

---

# 🔍 Expected Analysis

Particular attention will be given to visually similar chromosome classes.

The confusion matrix will help determine whether the model has difficulty distinguishing chromosomes with similar:

* Length
* Shape
* Centromere position
* Morphological structure
* Image intensity patterns

The performance of minority classes, especially **X and Y**, will also be analyzed separately.

---

# 🧬 Denver Classification

Human chromosomes can traditionally be grouped according to size and centromere position.

| Group | Chromosomes |
| ----- | ----------- |
| A     | 1, 2, 3     |
| B     | 4, 5        |
| C     | 6–12, X     |
| D     | 13, 14, 15  |
| E     | 16, 17, 18  |
| F     | 19, 20      |
| G     | 21, 22      |

Sex chromosomes:

```text
X
Y
```

The deep learning model, however, performs classification directly into the **24 individual classes**.

---

# 🏥 Potential Applications

The developed chromosome classification system can potentially support:

* 🧬 Cytogenetic analysis
* 🔬 Automated chromosome identification
* 🧪 Karyotype analysis assistance
* 🏥 Medical image analysis
* 🧫 Genetic disorder research
* 👨‍⚕️ Clinical decision-support systems
* 🔎 Computer-assisted chromosome analysis
* 🎓 Biomedical research and education

> The system is intended as a research/decision-support tool and not as a replacement for clinical diagnosis.

---

# 🚀 Future Work

The following improvements are planned:

* Complete ResNet18 optimization
* Train ResNet50
* Train InceptionV3
* Train EfficientNet
* Compare all architectures
* Optimize augmentation parameters
* Address Class X and Class Y imbalance
* Perform detailed confusion-matrix analysis
* Evaluate per-class performance
* Apply Grad-CAM for model explainability
* Analyze difficult/misclassified chromosome samples
* Optimize the best-performing model for inference
* Investigate ensemble learning after individual models are evaluated

---

# 🔬 Explainability

Grad-CAM will be used to visualize the regions that contribute most strongly to the model's predictions.

This can help answer:

> **"Which part of the chromosome did the model use to make its classification?"**

This is particularly useful for medical image analysis because it allows visual inspection of whether the model is focusing on meaningful chromosome structures rather than background artifacts.

---

# 📁 Project Structure

```text
chromosome-classification/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── resnet18_training.ipynb
│   ├── resnet50_training.ipynb
│   ├── inceptionv3_training.ipynb
│   └── efficientnet_training.ipynb
│
├── data/
│   ├── images/
│   └── labels/
│
├── preprocessing/
│   ├── crop
│   ├── clahe
│   ├── gaussian
│   └── sharpening
│
├── models/
│   ├── resnet18/
│   ├── resnet50/
│   ├── inceptionv3/
│   └── efficientnet/
│
├── outputs/
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── preprocessing_visualizations/
│   ├── classification_reports/
│   └── model_comparison/
│
└── README.md
```

---

# 💻 Technologies Used

* Python
* PyTorch
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn
* Pandas
* PIL
* Google Colab
* CUDA / GPU acceleration

---

# ▶️ How to Run

## Google Colab

1. Upload the project notebook to Google Colab.
2. Enable GPU acceleration.

```text
Runtime
   ↓
Change Runtime Type
   ↓
GPU
```

3. Load the chromosome dataset.
4. Run the preprocessing pipeline.
5. Create train/validation/test splits.
6. Apply augmentation to the training set.
7. Train the selected CNN architecture.
8. Evaluate the model.
9. Generate confusion matrices and classification reports.

---

# 📌 Current Project Status

```text
Dataset Collection                    ✅ Completed
24-Class Dataset Preparation          ✅ Completed
Train/Validation/Test Split           ✅ Completed
Sequential Preprocessing              ✅ Completed
Crop → CLAHE → Gaussian → Sharpening  ✅ Completed
Preprocessing Visualization            ✅ Completed
Initial ResNet18 Training              ✅ Completed
Baseline Evaluation                    ✅ Completed
Overfitting Analysis                   ✅ Completed

Stronger Data Augmentation             🔄 In Progress
Dropout / Regularization               🔄 In Progress
Class Weighting / Oversampling         🔄 In Progress
ResNet18 Optimization                  🔄 In Progress
ResNet50 Training                      ⏳ Planned
InceptionV3 Training                   ⏳ Planned
EfficientNet Training                  ⏳ Planned
Model Comparison                       ⏳ Planned
Grad-CAM Explainability                ⏳ Planned
Final Model Selection                  ⏳ Planned
```

---

# 🎯 Current Baseline

The current baseline demonstrates that the CNN can learn chromosome-specific features, achieving:

```text
Training Accuracy:       ~100%
Best Validation Accuracy: 82.10%
```

However, the large gap between training and validation accuracy demonstrates that the current model is overfitting.

The next stage of the project therefore focuses on **improving generalization through data augmentation, regularization, class balancing and transfer-learning strategies**, followed by comparison with ResNet50, InceptionV3 and EfficientNet.

---

# 👨‍💻 Internship Project

This project was developed as part of a **Summer Internship in Medical Image Analysis and Deep Learning**.

The work focuses on applying computer vision and deep learning techniques to automate chromosome image classification and investigate the challenges associated with small, imbalanced and visually similar medical image datasets.

---

# 📜 Acknowledgements

* Dataset provided as part of the internship/research work
* PyTorch pretrained architectures
* Open-source computer vision and machine learning libraries
* Medical image analysis research community

---

## ⭐ Key Highlights

* 🧬 **24-class chromosome classification**
* 🖼️ **5,474 single-chromosome images**
* 🔬 **Sequential image preprocessing**
* ⚙️ **Crop → CLAHE → Gaussian → Sharpening**
* 🔄 **Data augmentation for improved generalization**
* 🧠 **Deep CNN-based classification**
* 📊 **Confusion matrix and per-class evaluation**
* ⚖️ **Class imbalance analysis**
* 🔍 **Model explainability using Grad-CAM**
* 🚀 **Comparison of ResNet, Inception and EfficientNet architectures**
* 🏥 **Application in automated cytogenetic image analysis**
