# Garbage Image Classification: CNN with TensorFlow

A convolutional neural network (CNN) built from scratch to classify images of garbage into four material categories: brown glass, green glass, white glass, and plastic. The project involved rebuilding and correcting an existing pipeline, then iteratively improving model performance across six training runs.

## Results

| Model | Validation F1 | Notes |
|---|---|---|
| Baseline CNN | 0.873 | Provided reference model |
| Revised CNN | **0.850** | Final configuration after iterative tuning |

The baseline figure is included for reference. The revised pipeline applies proper data augmentation, a corrected 70/30 train/validation split, and integer-keyed class weights, all designed to give a more reliable estimate of real-world performance. The lower revised score reflects a harder, fairer evaluation rather than a regression in model quality.

An earlier iteration of this pipeline achieved a reported validation F1 of 0.913, but this figure was inflated. Augmentation was silently not being applied during training, and the validation set was drawn from the same unaugmented distribution. Correcting these issues produced a lower but more honest score of 0.850.

## Problem

Given a dataset of garbage images sourced from Kaggle ([Garbage Classification by Mostafa Abla](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)), the task was to classify images into four material categories. The dataset is moderately imbalanced across classes.

## Approach

### Data Preparation
- Loaded dataset from Google Drive (no API dependency)
- Filtered to the four relevant categories; removed remaining classes
- Resized and saved all images to 128×128 RGB via a preprocessing step
- Computed inverse-frequency class weights to account for class imbalance

### Data Augmentation
A single `ImageDataGenerator` instance handles both training and validation subsets. Augmentation is applied to the training subset only; the validation subset receives rescaling only, regardless of the augmentation parameters on the generator. This is an important distinction: using two separate generator instances with different `validation_split` values can cause train/validation overlap.

Augmentation parameters (training only):
- Rotation (±30°), zoom (±30%), width/height shifts (±15%), shear (15°)
- Horizontal and vertical flipping
- Brightness variation (0.7–1.3×) and channel shift (±20) to simulate varying lighting conditions

### Model Architecture
A sequential CNN with five convolutional blocks following the Conv2D → BatchNorm → ReLU → Pool pattern. Filter counts double across blocks (16 → 32 → 64 → 128 → 256).

Regularisation:
- L2 weight regularisation (`l2=0.0005`) on all Conv2D layers
- Dropout rates increase deeper in the network (0.3 → 0.4 → 0.5) where overfitting risk is higher
- `GlobalAveragePooling2D` used instead of `Flatten` to reduce parameter count

### Training
- Optimiser: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Metrics: Custom F1 score (precision/recall via Keras backend)
- Callbacks:
  - `EarlyStopping`: monitors `val_loss`, patience=10
  - `ReduceLROnPlateau`: halves LR on `val_loss` plateau, patience=5, min_lr=1e-5
  - `ModelCheckpoint`: saves best weights by `val_f1_m` (mode=max)
- Train/validation split: 70/30

### Iterative Improvement
Performance was improved systematically across six runs by addressing specific issues identified in each run:

| Change | Effect |
|---|---|
| Corrected augmentation pipeline (single generator) | Resolved augmentation being silently skipped on validation |
| Increased L2 regularisation (0.0002 → 0.0005) | Reduced train/val gap from ~9pts to ~7pts |
| Increased dropout (0.1/0.2/0.3 → 0.3/0.4/0.5) | Further reduced overfitting |
| Added brightness/channel shift augmentation | Improved generalisation to lighting variation |
| Raised `ReduceLROnPlateau` `patience` (3 → 5) | Prevented premature LR decay on noisy early gradients |
| Reduced batch size (64 → 32) | More gradient updates per epoch; marginal generalisation gain |
| Switched `ModelCheckpoint` to monitor `val_f1_m` | Saved model matches reported metric |

### Evaluation
- Confusion matrix
- Per-class F1, precision, and recall via classification report
- Visualisation of misclassified examples

## Limitations & Next Steps

The ~6pt train/val gap (train F1 ~0.91, val F1 ~0.85) indicates moderate overfitting that proved resistant to further regularisation, consistent with a relatively small dataset. Transfer learning (e.g. MobileNetV2 or EfficientNetB0) would be the natural next step and is planned as a separate notebook.

## Tech Stack

- Python 3.11
- TensorFlow 2.18 / Keras
- NumPy
- scikit-learn (metrics)
- Matplotlib, Seaborn
- Google Colab (L4 GPU)

## Files

- `Garbage_Classification_CNN.ipynb`: Full notebook including data preparation, model definition, training, and evaluation
