# Garbage Image Classification — CNN with TensorFlow

A convolutional neural network (CNN) built to classify images of garbage into material categories (brown glass, green glass, white glass, and plastic). The goal was to improve on a provided baseline model by applying regularisation techniques and improved data augmentation.

## Results

| Model | Validation F1 Score |
|---|---|
| Baseline CNN | 0.8730 |
| Improved CNN | **0.9139** |

## Problem

Given a dataset of garbage images sourced from Kaggle ([Garbage Classification by Mostafa Abla](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)), the task was to classify images into four material categories. The dataset is moderately imbalanced across classes.

## Approach

### Data Preparation
- Downloaded dataset via the Kaggle API
- Filtered to the four relevant categories
- Resized and preprocessed images to 128x128 RGB
- Applied custom class weights to account for class imbalance

### Data Augmentation
Extended the baseline augmentation with additional parameters to improve generalisation:
- Rotation, zoom, width/height shifts, shear
- Horizontal and vertical flipping

### Model Architecture
A sequential CNN with three convolutional blocks, each containing:
- Conv2D layer with L2 regularisation
- Batch Normalisation
- ReLU Activation
- Max Pooling

Followed by Global Average Pooling and a Dense output layer with softmax activation.

### Training
- Optimiser: Adam
- Loss: Categorical Crossentropy
- Metrics: Custom F1 score (precision/recall implementation via Keras backend)
- Callbacks: Early Stopping, ReduceLROnPlateau, ModelCheckpoint (best weights saved)
- Train/validation split: 80/20 with stratification

### Evaluation
- Confusion matrix
- Per-class F1 scores
- Classification report (precision, recall, F1 per class)
- Visualisation of misclassified examples

## Tech Stack

- Python 3.11
- TensorFlow 2.18 / Keras
- NumPy, Pandas
- scikit-learn (train/test split, label encoding, PCA, StandardScaler, metrics)
- Matplotlib, Seaborn
- Google Colab (L4 GPU)

## Files

- `Garbage_Classification_CNN.ipynb` — Full notebook including data preparation, model definition, training, and evaluation
