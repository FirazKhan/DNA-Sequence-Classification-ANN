import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from warnings import filterwarnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [15,8]
# Load the dataset
df = pd.read_csv("C:/Users/moham/OneDrive/Desktop/Msc/Machine Learning/Part 1/dna.csv")
df.head()
# Display info about the DataFrame
df.info()
# 1. Handling Missing Data
# Check for missing values in the dataset
print("Total missing values:", df.isnull().sum().sum())
# There are no missing values in the dataset, so no imputation or removal is necessary.
df.dtypes
# Identify the class/label column
class_col = 'class'  # Target column

# Select feature columns (exclude class/label)
feature_cols = [col for col in df.columns if col != class_col]
# 2. Label Encoding
# Apply label encoding to the target column (class) to convert 1,2,3 to 0,1,2
le = LabelEncoder()
df['class_encoded'] = le.fit_transform(df['class'])
print(df[['class_encoded']].head())
# Remove the original 'class' column after encoding
df = df.drop('class', axis=1)
# 3. Normalisation
# Not performed because all features are already binary (0/1), so normalization is not required.
# 4. Splitting the Data
# Split the cleaned dataset into training and testing sets (80/20 split), using stratified sampling to maintain class distribution
x = df.drop('class_encoded', axis=1)
y = df['class_encoded']

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", xtrain.shape)
print("Testing set shape:", xtest.shape)
print("Training class distribution:\n", ytrain.value_counts(normalize=True))
print("Testing class distribution:\n", ytest.value_counts(normalize=True))
# --- Deep Learning Model: MLP (ANN) for Classification ---
# Import required libraries for deep learning
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import time
import itertools
# 1. Define the ANN architecture
def build_model(n_hidden=2, n_neurons=64, learning_rate=0.001, activation='relu', dropout_rate=0.2):
    """
    Build a feedforward neural network for DNA sequence classification.
    
    Parameters:
    - n_hidden: number of hidden layers
    - n_neurons: number of neurons in each hidden layer
    - learning_rate: learning rate for the optimizer
    - activation: activation function for hidden layers
    - dropout_rate: dropout rate for regularization
    
    Returns:
    - Compiled Keras model
    """
    # Create the model
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(x.shape[1],)))  # 180 features
    
    # Hidden layers
    for _ in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation=activation))
        model.add(layers.BatchNormalization())  # Add batch normalization
        model.add(layers.Dropout(dropout_rate))  # Add dropout for regularization
    
    # Output layer (3 classes: EI, IE, Neither)
    model.add(layers.Dense(3, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 2. Implement Manual Hyperparameter Optimization
print("\nStarting Hyperparameter Optimization:")

# Define the hyperparameter search space
param_grid = {
    'n_hidden': [1, 2],
    'n_neurons': [64, 128],
    'learning_rate': [0.001, 0.01],
    'activation': ['relu'],
    'dropout_rate': [0.2, 0.3],
    'batch_size': [32, 64]
}

# Initialize K-fold cross-validation
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store best results
best_params = None
best_accuracy = 0
best_model = None
all_results = []

# Generate all parameter combinations
param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
total_combinations = len(param_combinations)

print(f"\nTesting {total_combinations} parameter combinations...")

# Test each parameter combination
for i, params in enumerate(param_combinations, 1):
    print(f"\nTesting combination {i}/{total_combinations}")
    print("Parameters:", params)
    
    # Store results for this combination
    fold_scores = []
    fold_times = []
    
    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(x, y), 1):
        # Split data
        x_train_fold = x.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        x_val_fold = x.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Build and train model
        model = build_model(
            n_hidden=params['n_hidden'],
            n_neurons=params['n_neurons'],
            learning_rate=params['learning_rate'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )

        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Record training time
        start_time = time.time()
        
        # Train the model
        model.fit(
            x_train_fold, y_train_fold,
            validation_data=(x_val_fold, y_val_fold),
            epochs=100,  
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        fold_times.append(training_time)
        
        # Evaluate model
        _, val_accuracy = model.evaluate(x_val_fold, y_val_fold, verbose=0)
        fold_scores.append(val_accuracy)
    
    # Calculate mean accuracy for this combination
    mean_accuracy = np.mean(fold_scores)
    mean_time = np.mean(fold_times)
    
    # Store results
    result = {
        'params': params,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': np.std(fold_scores),
        'mean_time': mean_time
    }
    all_results.append(result)
    
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"Mean Training Time: {mean_time:.2f} seconds")
    
    # Update best parameters if this combination is better
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = params
        best_model = model

# Sort results by accuracy
all_results.sort(key=lambda x: x['mean_accuracy'], reverse=True)

# Print optimization results
print("\n" + "="*50)
print("Hyperparameter Optimization Results:")
print("="*50)
print("\nTop 3 Parameter Combinations:")
for i, result in enumerate(all_results[:3], 1):
    print(f"\n{i}. Parameters:")
    for param, value in result['params'].items():
        print(f"   {param}: {value}")
    print(f"   Mean Accuracy: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
    print(f"   Mean Training Time: {result['mean_time']:.2f} seconds")

print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# 3. Train final model with best parameters
print("\nTraining final model with best parameters...")

# Initialize K-fold cross-validation
fold_scores = []
fold_histories = []
fold_times = []
best_fold = 0
best_accuracy = 0
best_fold_model = None  # Store the best model from the best fold

# Perform k-fold cross validation with best parameters
for fold, (train_idx, val_idx) in enumerate(kfold.split(x, y), 1):
    print(f"\n{'='*50}")
    print(f"Fold {fold}/{n_splits}")
    print(f"{'='*50}")
    
    # Split data
    xtrain_fold = x.iloc[train_idx]
    ytrain_fold = y.iloc[train_idx]
    xval_fold = x.iloc[val_idx]
    yval_fold = y.iloc[val_idx]
    
    # Print fold information
    print(f"Training samples: {len(xtrain_fold)}")
    print(f"Validation samples: {len(xval_fold)}")
    print("\nClass distribution in training set:")
    print(y_train_fold.value_counts(normalize=True))
    
    # Build and train model
    model = build_model(
        n_hidden=best_params['n_hidden'],
        n_neurons=best_params['n_neurons'],
        learning_rate=best_params['learning_rate'],
        activation=best_params['activation'],
        dropout_rate=best_params['dropout_rate']
    )

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Record training time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        xtrain_fold, ytrain_fold,
        validation_data=(xval_fold, yval_fold),
        epochs=100,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    fold_times.append(training_time)
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(xval_fold, yval_fold, verbose=0)
    fold_scores.append(val_accuracy)
    fold_histories.append(history.history)
    
    # Update best model tracking
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_fold = fold
        best_fold_model = model  # Save the best model
    
    print(f"\nFold {fold} Results:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

# Calculate and print comprehensive results
mean_accuracy = np.mean(fold_scores)
std_accuracy = np.std(fold_scores)
mean_time = np.mean(fold_times)

print("\n" + "="*50)
print("Final Results with Best Parameters:")
print("="*50)
print("Best Parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")
print(f"\nMean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
print(f"Best Accuracy: {best_accuracy:.4f} (Fold {best_fold})")
print(f"Average Training Time per Fold: {mean_time:.2f} seconds")
print(f"Total Training Time: {sum(fold_times):.2f} seconds")

# 4. Print model summary
print("\nModel Architecture Summary:")
best_fold_model.summary()  # Use the best model for summary

# 5. Explain the architecture choices
print("\nArchitecture Design Choices:")
print("1. Input Layer:")
print(f"   - Shape: {x.shape[1]} (number of features)")
print("2. Hidden Layers:")
print(f"   - Number of layers: {best_params['n_hidden']}")
print(f"   - Neurons per layer: {best_params['n_neurons']}")
print(f"   - Activation: {best_params['activation']}")
print("   - Batch Normalization: Yes")
print(f"   - Dropout: {best_params['dropout_rate']}")
print("3. Output Layer:")
print("   - Neurons: 3 (one for each class)")
print("   - Activation: Softmax (for multi-class classification)")
print("4. Training Configuration:")
print(f"   - Optimizer: Adam (learning rate: {best_params['learning_rate']})")
print("   - Loss Function: Sparse Categorical Crossentropy")
print("   - Metrics: Accuracy")
print("5. Cross-validation Configuration:")
print(f"   - Number of folds: {n_splits}")
print(f"   - Mean Accuracy: {mean_accuracy:.4f}")
print(f"   - Standard Deviation: {std_accuracy:.4f}")
print(f"   - Best Fold: {best_fold}")
print(f"   - Epochs: 100")
print(f"   - Batch Size: {best_params['batch_size']}")
print(f"   - Average Training Time: {mean_time:.2f} seconds")

# 6. Save the best model
print("\nSaving the best model...")
best_fold_model.save('ANN-model.h5')

# 7. Evaluate the best model on the test set
print("\nEvaluating best model on test set...")
test_loss, test_accuracy = best_fold_model.evaluate(xtest, ytest, verbose=0)
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Test Set Loss: {test_loss:.4f}")

# 8. Visualizations

# Plot training and validation curves for the best fold
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(fold_histories[best_fold-1]['accuracy'], label='Training Accuracy')
plt.plot(fold_histories[best_fold-1]['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Best Fold)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(fold_histories[best_fold-1]['loss'], label='Training Loss')
plt.plot(fold_histories[best_fold-1]['val_loss'], label='Validation Loss')
plt.title('Model Loss (Best Fold)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Create activation heatmap
def get_layer_activations(model, input_data):
    """Get activations for the first hidden layer"""
    # Create a new model that outputs the activations of the first hidden layer
    activation_model = keras.Model(
        inputs=model.inputs,
        outputs=model.layers[1].output  # First hidden layer
    )
    # Get activations
    activations = activation_model.predict(input_data)
    return activations

# Get activations for the first hidden layer
print("Generating activation heatmap...")
activations = get_layer_activations(best_fold_model, xtest)

# Plot activation heatmap
plt.figure(figsize=(15, 5))

# Plot mean activations across neurons
plt.subplot(1, 2, 1)
mean_activations = np.mean(activations, axis=0)
plt.bar(range(len(mean_activations)), mean_activations)
plt.title('Mean Neuron Activations')
plt.xlabel('Neuron')
plt.ylabel('Mean Activation')
plt.grid(True)

# Plot activation heatmap
plt.subplot(1, 2, 2)
plt.imshow(activations[:100].T, aspect='auto', cmap='viridis')  # Show first 100 samples
plt.colorbar(label='Activation')
plt.title('Neuron Activation Heatmap')
plt.xlabel('Sample')
plt.ylabel('Neuron')
plt.tight_layout()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Get predictions
ypred = best_fold_model.predict(xtest)
ypred_classes = np.argmax(ypred, axis=1)

# Create confusion matrix
cm = confusion_matrix(ytest, ypred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Get predictions
ypred = best_fold_model.predict(xtest)
ypred_classes = np.argmax(ypred, axis=1)

# Create confusion matrix
cm = confusion_matrix(ytest, ypred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(range(len(ytrain)), ytrain, label='Actual', alpha=0.6)
plt.scatter(range(len(ytrain)), ytrain_pred_classes, label='Predicted', alpha=0.6)
plt.title('Train: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Class')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(range(len(ytest)), ytest, label='Actual', alpha=0.6)
plt.scatter(range(len(ytest)), ytest_pred_classes, label='Predicted', alpha=0.6)
plt.title('Test: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Class')
plt.legend()
plt.tight_layout()