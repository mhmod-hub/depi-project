# Microscope AI — Cell Segmentation & Deployment
DEPI Graduation Project | Human Protein Atlas | UNet ResNet34

## What This Does
End-to-end AI pipeline for fluorescence microscopy image segmentation.
Segments cell images into background, cell body, and nucleus at 92% accuracy.
Built for drug discovery automation.

## Live Demo
https://huggingface.co/spaces/Jimmy10/microscope-ai-live-profiler

## How To Run
This notebook is designed to run on Kaggle, not locally.
The dataset is 19GB and is available natively on Kaggle.

1. Go to Kaggle and open the notebook:
  https://www.kaggle.com/code/jome1g/microscope-ai-segmentation-and-deployment
2. Attach the dataset:
   Human Protein Atlas Image Classification
   kaggle.com/c/human-protein-atlas-image-classification
3. Set accelerator to GPU T4 x2
4. Turn on Internet
5. Run All

## Results
| Class | F1 Score |
|-------|----------|
| Background | 0.94 |
| Cell Body | 0.81 |
| Nucleus | 0.93 |
| **Overall Accuracy** | **92%** |

## Tech Stack
PyTorch · PyTorch Lightning · UNet ResNet34 · 
ONNX · Streamlit · FastAPI · HuggingFace Spaces · Kaggle

## Team
Mahmoud Refaat

Mazen Mohamed 

Abdulrahman Jamal

Mazen Hassan

Hossam Mostafa

DEPI — Digital Egypt Pioneers Initiative
