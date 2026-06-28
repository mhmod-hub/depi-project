# Microscope AI — Cell Segmentation & Deployment

## Introduction

In modern pharmaceutical research and drug discovery, understanding how drugs affect cells at a structural level is a fundamental requirement. Fluorescence microscopy has become one of the most powerful tools for visualizing protein localization and cellular morphology — yet the sheer volume of images produced by large-scale drug screening experiments far exceeds what human experts can manually analyze. A single compound screening campaign can generate hundreds of thousands of microscopy images, each requiring expert-level annotation to extract meaningful biological insights.

To address this challenge, we developed an end-to-end AI pipeline for automated cell segmentation and biological profiling. By training a deep learning model on real fluorescence microscopy data, we enable the automatic labeling of every pixel in a cell image as either background, cell body, or nucleus — producing structured, quantitative outputs that researchers can use to measure drug-induced changes at scale.

---

## Problem Statement

Drug discovery researchers currently face a critical bottleneck: manually analyzing microscopy images to assess how chemical compounds affect cellular structure is slow, expensive, and inconsistent across different analysts. Human resources teams in pharmaceutical labs rely on trained biologists to manually delineate cells and nuclei, a process that is not feasible at the scale required by modern high-throughput drug screens.

The objective of this project is to develop an automated AI segmentation pipeline capable of accurately identifying and labeling three biological regions — background, cell body, and nucleus — in 4-channel fluorescence microscopy images. This pipeline provides quantitative per-image metrics (cell coverage, nucleus coverage, cell-to-nucleus ratio) that enable researchers to detect structural changes caused by drug compounds automatically, consistently, and at scale.

---

## Dataset

**Source:** Human Protein Atlas (HPA) Image Classification — Kaggle Competition

**General Overview**

| Property | Value |
|----------|-------|
| Total images (full dataset) | ~31,000 |
| Training subset used | 6,000 images |
| Validation subset used | 1,500 images |
| Total dataset size | ~19 GB |
| Image format | Grayscale PNG per channel |
| Channels per image | 4 (Blue, Red, Yellow, Green) |
| Missing values | None |

**Dataset Structure**

Each sample in the dataset consists of four separate grayscale PNG images, each captured under a different fluorescence channel:

- **Blue channel** — Nuclei (DAPI stain)
- **Red channel** — Microtubules
- **Yellow channel** — Endoplasmic Reticulum
- **Green channel** — Target Protein (varies per sample)

Each image is associated with one or more protein localization labels from 28 possible classes (e.g., nucleus, cytoplasm, mitochondria), making the original competition a multi-label classification problem. Our project repurposes these images for pixel-level semantic segmentation, generating 3-class masks (background, cell body, nucleus) via automated Otsu thresholding on the blue and red channels.

**Label Distribution**

The dataset exhibits significant class imbalance at the pixel level, which is expected in microscopy data:
- Background pixels dominate each image (empty space between cells)
- Cell body pixels form the second largest class
- Nucleus pixels are the smallest class but the most visually distinct

---

## Methodology

### Preprocessing & Cache Pipeline

One of the key performance challenges in this project was the scale of the dataset. Raw preprocessing — including CLAHE (Contrast Limited Adaptive Histogram Equalization) per channel, min-max normalization, and Otsu thresholding for mask generation — was computationally expensive when applied live during training. To solve this, we implemented an offline preprocessing cache:

All 7,500 images in our training and validation subsets were preprocessed once and saved as compressed `.npz` files to disk. During training, the dataset class simply loads these cached files instead of recomputing preprocessing on every sample every epoch. This reduced per-epoch data loading overhead dramatically and was the primary factor in bringing training time within feasible limits on Kaggle's T4 GPU.

### Dataset Splitting

A **5-Fold Stratified Cross-Validation** split was applied to the full dataset before subsetting. Stratification was performed on the first protein localization label per image to ensure each fold maintained a representative class distribution. Fold 0 was designated as the validation set; folds 1–4 formed the training pool. A random stratified sample of 6,000 training and 1,500 validation images was then drawn from the respective folds.

### Model Architecture

The segmentation model is a **UNet with a pretrained ResNet34 encoder**, implemented via the `segmentation-models-pytorch` library. UNet is the standard architecture for biomedical image segmentation, using an encoder-decoder structure with skip connections to preserve spatial detail during upsampling.

A key architectural challenge was that the pretrained ResNet34 encoder expects 3-channel RGB input, while our data has 4 fluorescence channels. We resolved this with a custom **first-layer adaptation**: the original 3-channel convolutional weights were retained, and a 4th channel weight was added by computing the mean of the existing 3 channels. This allowed us to leverage ImageNet pretrained weights while accepting 4-channel input, preserving transfer learning benefits without retraining from scratch.

**Model output:** 3-class per-pixel prediction (0 = Background, 1 = Cell Body, 2 = Nucleus)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch Lightning |
| Loss function | Cross-Entropy Loss |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Precision | 16-mixed (fp16) |
| Batch size | 32 |
| Input resolution | 256 × 256 |
| Epochs | 3 |
| Early stopping patience | 1 epoch |
| Metric tracked | Validation Loss |

### Evaluation Metrics

Model performance was evaluated using pixel-level metrics across all three classes:
- **Accuracy** — percentage of correctly labeled pixels overall
- **Precision** — of all pixels predicted as a class, how many were correct
- **Recall** — of all actual pixels of a class, how many were detected
- **F1-Score** — harmonic mean of precision and recall

---

## Results

The trained UNet model was evaluated on 1,500 held-out validation images. Pixel-level predictions were compared against ground truth masks generated by Otsu thresholding.

### Pixel-Level Classification Report

| Class | Precision | Recall | F1-Score | Support (pixels) |
|-------|-----------|--------|----------|------------------|
| Background (0) | 0.94 | 0.94 | 0.94 | 68,206,785 |
| Cell Body (1) | 0.83 | 0.80 | 0.81 | 19,258,674 |
| Nucleus (2) | 0.89 | 0.97 | 0.93 | 10,838,541 |
| **Overall Accuracy** | | | **0.92** | 98,304,000 |
| Macro Average | 0.89 | 0.90 | 0.89 | |
| Weighted Average | 0.92 | 0.92 | 0.92 | |

### Key Observations

**Background segmentation (F1: 0.94)** is near-perfect. The model reliably distinguishes empty space from biological structures, which is important for accurate cell counting and coverage measurement.

**Nucleus segmentation (F1: 0.93)** is excellent, achieving a recall of 0.97 — meaning the model detects 97% of all nucleus pixels. Nuclei are visually compact and bright in the blue channel, making them the most learnable structure.

**Cell body segmentation (F1: 0.81)** is the weakest class, with 18% of cell body pixels misclassified as background. This is an expected and explainable result: cell boundaries are thin, diffuse structures that lose spatial detail when images are downscaled to 256×256 for training. This is a known challenge in fluorescence microscopy segmentation and is the primary target for future improvement.

---

## Deployment

The trained model was exported to **ONNX format** (Open Neural Network Exchange) for production inference, enabling fast, framework-independent predictions without requiring PyTorch at runtime.

A complete deployment stack was built and automated:

- **FastAPI backend** — REST API accepting 4 channel images and returning predicted masks as JSON
- **Streamlit frontend** — web UI for uploading 4-channel images and visualizing segmentation results with statistics
- **Docker container** — packages the app and dependencies for consistent deployment
- **Hugging Face Spaces** — hosts the live application publicly at no cost

The entire deployment pipeline is automated within the notebook — running the final cells generates all application files and deploys them to Hugging Face without any manual steps.

**Live application:** https://huggingface.co/spaces/Jimmy10/microscope-ai-live-profiler

---

## Conclusion

This project successfully demonstrates the feasibility of automated fluorescence microscopy cell segmentation using deep learning. By training a 4-channel adapted UNet ResNet34 on a representative subset of the Human Protein Atlas dataset, we achieved 92% overall pixel accuracy with strong performance across all three biological classes.

The pipeline addresses a genuine bottleneck in drug discovery — the manual analysis of large-scale microscopy experiments — by providing fast, consistent, and scalable cell segmentation that produces structured quantitative outputs researchers can act on. What previously required hours of expert manual annotation can now be processed in seconds through our deployed web application.

The full pipeline — from raw fluorescence images to deployed live segmentation tool — is contained within a single reproducible Kaggle notebook, making it accessible and extensible for future research teams.

---

## Future Work

To further enhance the biological and technical value of this project, future iterations could focus on the following:

**Before/After Drug Comparison Engine:** Extend the Streamlit application to accept two sets of images (control and treated) and automatically generate a structured comparison report showing percentage changes in cell coverage, nucleus size, and cell-to-nucleus ratio — directly enabling quantitative drug effect assessment without any manual measurement.

**Improved Cell Body Segmentation:** The primary quality gap is cell body F1 at 0.81. Increasing training resolution from 256×256 to 512×512, training for more epochs, adding Dice Loss to the hybrid loss function, and incorporating richer data augmentation (flips, rotations, brightness contrast) are all well-established techniques that would directly address the boundary detection weakness.

**Full Dataset Training:** Our current results use only 7,500 of the available ~31,000 images due to compute constraints. Scaling to the full dataset with adequate GPU resources would meaningfully improve generalization across the 28 protein localization classes represented in the HPA dataset.

**Explainable AI (XAI):** Implement Grad-CAM or attention map visualization to show which regions of the input image most influenced the model's segmentation decisions, making the model's behavior interpretable to non-technical researchers.

**SHAP-based Protein Feature Analysis:** Connect segmentation outputs to downstream protein localization analysis, enabling the model to not only segment cells but flag which protein distribution patterns deviate from baseline — directly linking cell morphology changes to molecular-level drug effects.
