# DNA Sequence Classification using Artificial Neural Networks

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Project Overview

This project implements a deep learning solution for **DNA sequence classification** using Artificial Neural Networks (ANNs). The model classifies DNA sequences into three categories: **Exon-Intron (EI)**, **Intron-Exon (IE)**, and **Neither**, achieving an impressive **95%+ accuracy** through systematic hyperparameter optimization and k-fold cross-validation.

This was developed as part of the **ECMM422J Machine Learning Coursework** at the University of Exeter.

## 🎯 Key Features

- **Multi-class Classification**: Classifies DNA sequences into 3 distinct splice junction categories
- **Automated Hyperparameter Tuning**: Tests 32 different parameter combinations systematically
- **K-Fold Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation
- **Regularization Techniques**: Implements dropout and batch normalization to prevent overfitting
- **Model Persistence**: Saves the best-performing model for future predictions
- **Comprehensive Visualization**: Training curves, confusion matrices, and activation heatmaps

## 📊 Dataset

- **Source**: DNA splice junction dataset
- **Samples**: 3,186 DNA sequences
- **Features**: 180 binary features (representing nucleotide positions)
- **Classes**: 3 categories
  - Class 0: Intron-Exon (IE) boundaries
  - Class 1: Exon-Intron (EI) boundaries
  - Class 2: Neither (no splice junction)
- **Class Distribution**: Imbalanced (~52% class 2, ~24% each for classes 0 and 1)

## 🧠 Model Architecture

### Best Performing Configuration

```
Input Layer: 180 features
  ↓
Hidden Layer: 128 neurons
  - Activation: ReLU
  - Batch Normalization
  - Dropout: 0.3
  ↓
Output Layer: 3 neurons (Softmax)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 1 |
| Neurons per Layer | 128 |
| Learning Rate | 0.001 |
| Activation Function | ReLU |
| Dropout Rate | 0.3 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

## 📈 Performance Metrics

### Cross-Validation Results
- **Mean Accuracy**: 95.64% (± 0.62%)
- **Best Fold Accuracy**: 95.76%
- **Average Training Time**: ~5 seconds per fold

### Test Set Performance
- **Test Accuracy**: **99.53%**
- **Test Loss**: 0.0217

### Per-Class Metrics (Test Set)

| Class | Precision | Recall | F1-Score | Specificity |
|-------|-----------|--------|----------|-------------|
| 0 (IE) | 0.99 | 1.00 | 1.00 | 0.998 |
| 1 (EI) | 0.99 | 0.99 | 0.99 | 0.998 |
| 2 (Neither) | 1.00 | 0.99 | 1.00 | 0.997 |

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dna-classification-ann.git
cd dna-classification-ann

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Training the Model

```python
# Run the Jupyter notebook
jupyter notebook Training-ANN.ipynb
```

Or use the Python script:

```python
python "Ml 1.py"
```

#### Loading the Pre-trained Model

```python
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('ANN-model.h5')

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

## 📁 Project Structure

```
Part 1/
│
├── Training-ANN.ipynb              # Main Jupyter notebook with full implementation
├── Ml 1.py                          # Python script version
├── dna.csv                          # DNA sequence dataset
├── ANN-model.h5                     # Saved trained model
├── Thameem Ansari-Mohammed Firaz-750011330.pdf  # Technical report
├── ref.bib                          # Bibliography references
├── lat.sty                          # LaTeX style file
└── README.md                        # This file
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Check**: Verified no missing data
- **Label Encoding**: Converted class labels (1,2,3) to (0,1,2)
- **No Normalization**: Features are already binary (0/1)
- **Stratified Split**: 80/20 train-test split maintaining class distribution

### 2. Hyperparameter Optimization
- **Search Space**: 32 parameter combinations tested
- **Cross-Validation**: 5-fold stratified k-fold
- **Early Stopping**: Patience of 10 epochs on validation accuracy
- **Criteria**: Selected model with highest mean cross-validation accuracy

### 3. Model Training
- **Training Strategy**: K-fold cross-validation with the best parameters
- **Regularization**: Dropout (0.3) and Batch Normalization
- **Epochs**: Up to 100 with early stopping
- **Optimization**: Adam optimizer with learning rate 0.001

### 4. Evaluation
- Confusion matrix analysis
- Per-class precision, recall, F1-score, and specificity
- Training/validation curves
- Activation heatmaps

## 📊 Visualizations

The notebook generates several visualizations:

1. **Training Curves**: Accuracy and loss over epochs
2. **Confusion Matrix**: Classification performance breakdown
3. **Activation Heatmap**: First hidden layer neuron activations
4. **Prediction Scatter Plots**: Actual vs predicted classes

## 🎓 Key Insights

1. **Model Generalization**: Minimal gap between training and validation accuracy indicates excellent generalization
2. **Early Convergence**: Model reaches optimal performance within 20-40 epochs
3. **Regularization Effectiveness**: Dropout rate of 0.3 effectively prevents overfitting
4. **Architecture Simplicity**: A single hidden layer is sufficient for this classification task
5. **Class Balance**: Stratified sampling ensures proper representation of all classes

## 📚 References

- Lapedes, A., et al. (1988). "Use of adaptive networks for detection of genomic DNA functional sites"
- California Housing Dataset documentation
- TensorFlow/Keras documentation

## 📧 Contact

For questions or collaborations, please open an issue or contact through GitHub.

---

**Note**: This project was completed as part of academic coursework. The implementation follows best practices in deep learning and includes comprehensive documentation for reproducibility.

