"""ORIGINAL training script, kept for transparency only.
It uses the test split as validation data. The results reported in the paper
come from experiments/rerun_experiments.py, which uses a separate validation set
and five seeds."""
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_DIR = "../data/synthetic_dataset_robust"
CLASSES = ["Healthy", "Intermittent", "Constant"]
SEQUENCE_LENGTH = 150  # 5 seconds * 30 fps
FEATURES = 2           # Left Ratio, Right Ratio

# --- 1. Load Data ---
X = [] # Signals
y = [] # Labels

print("Loading dataset...")

for label_name in CLASSES:
    path = os.path.join(DATA_DIR, label_name, "*.csv")
    files = glob.glob(path)
    
    for file in files:
        df = pd.read_csv(file)
        # We only care about the normalized ratios, not raw timestamps
        signal = df[['L_ratio', 'R_ratio']].values
        
        # Safety check: Ensure length is correct
        if len(signal) == SEQUENCE_LENGTH:
            X.append(signal)
            y.append(label_name)

X = np.array(X)
y = np.array(y)

# Encode labels (Healthy=0, Intermittent=1, Constant=2)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

# --- 2. Build the Model (CNN-BiLSTM) ---
model = Sequential([
    # Layer 1: 1D Convolution to extract features from the signal curve
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURES)),
    MaxPooling1D(pool_size=2),
    
    # Layer 2: Dropout to prevent overfitting
    Dropout(0.3),
    
    # Layer 3: Bidirectional LSTM to understand temporal patterns
    Bidirectional(LSTM(64, return_sequences=False)),
    
    # Layer 4: Dense Classifier
    Dense(32, activation='relu'),
    
    # Output Layer: 3 neurons (one for each class), Softmax for probability
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\n--- Starting Training ---")
history = model.fit(
    X_train, y_train,
    epochs=20,             # Iterations over the dataset
    batch_size=32,
    validation_data=(X_test, y_test)
)

# --- 4. Save the Brain ---
model.save("dynamic_hirschberg_model.h5")
print("\nModel saved as 'dynamic_hirschberg_model.h5'")

# --- 5. Quick Evaluation ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")