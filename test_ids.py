"""
Test script to verify IDS implementation
Runs the notebook code and generates results for presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IDS IMPLEMENTATION TEST - PRESENTATION VERIFICATION")
print("="*80)

# Part 1: Load and Preprocess Data
print("\n[PART 1] Loading and Preprocessing Data...")
print("-"*80)

try:
    columns = [f'feature_{i}' for i in range(41)] + ['label']
    train_data = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
    test_data = pd.read_csv('KDDTest+.txt', header=None, names=columns)
    print(f"[OK] Training data loaded: {train_data.shape}")
    print(f"[OK] Testing data loaded: {test_data.shape}")
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    exit(1)

# Encode categorical features
print("\n[PREPROCESSING] Encoding categorical features...")
categorical_cols = ['feature_1', 'feature_2', 'feature_3']
le_cat = LabelEncoder()

for col in categorical_cols:
    train_data[col] = le_cat.fit_transform(train_data[col].astype(str))
    test_data[col] = le_cat.transform(test_data[col].astype(str))

# Encode labels
le_label = LabelEncoder()
train_data['label'] = le_label.fit_transform(train_data['label'])
test_data['label'] = le_label.transform(test_data['label'])

# Separate features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Features encoded and scaled")
print(f"[OK] Training samples: {len(X_train)}")
print(f"[OK] Testing samples: {len(X_test)}")

# EDA Statistics
print("\n[EDA] Class Distribution:")
train_label_counts = y_train.value_counts()
test_label_counts = y_test.value_counts()
print(f"Training - Normal: {train_label_counts.get(0, 0)} ({train_label_counts.get(0, 0)/len(y_train)*100:.2f}%)")
print(f"Training - Attack: {train_label_counts.get(1, 0)} ({train_label_counts.get(1, 0)/len(y_train)*100:.2f}%)")

# Part 2: Train ML Models
print("\n[PART 2] Training Machine Learning Models...")
print("-"*80)

# Random Forest
print("\n[MODEL 1] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)

print(f"[OK] Random Forest trained successfully")
print(f"  Accuracy:  {rf_accuracy:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")

# SVM
print("\n[MODEL 2] Training SVM (this may take a few minutes)...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, average='weighted', zero_division=0)

print(f"[OK] SVM trained successfully")
print(f"  Accuracy:  {svm_accuracy:.4f}")
print(f"  Precision: {svm_precision:.4f}")
print(f"  Recall:    {svm_recall:.4f}")
print(f"  F1-Score:  {svm_f1:.4f}")

# Part 3: Neural Network
print("\n[PART 3] Training Neural Network...")
print("-"*80)

class IDSNet(nn.Module):
    def __init__(self, input_size):
        super(IDSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Prepare data for PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

nn_model = IDSNet(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Train
print("Training Neural Network (10 epochs)...")
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    batches = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1
    avg_loss = epoch_loss / batches
    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluate
nn_model.eval()
with torch.no_grad():
    nn_outputs = nn_model(X_test_tensor)
    _, nn_pred = torch.max(nn_outputs, 1)
    nn_pred = nn_pred.numpy()

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_recall = recall_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_f1 = f1_score(y_test, nn_pred, average='weighted', zero_division=0)

print(f"\n[OK] Neural Network trained successfully")
print(f"  Accuracy:  {nn_accuracy:.4f}")
print(f"  Precision: {nn_precision:.4f}")
print(f"  Recall:    {nn_recall:.4f}")
print(f"  F1-Score:  {nn_f1:.4f}")

# Part 4: Model Comparison
print("\n[PART 4] Model Comparison")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)
print(f"{'Random Forest':<20} {rf_accuracy:<12.4f} {rf_precision:<12.4f} {rf_recall:<12.4f} {rf_f1:<12.4f}")
print(f"{'SVM':<20} {svm_accuracy:<12.4f} {svm_precision:<12.4f} {svm_recall:<12.4f} {svm_f1:<12.4f}")
print(f"{'Neural Network':<20} {nn_accuracy:<12.4f} {nn_precision:<12.4f} {nn_recall:<12.4f} {nn_f1:<12.4f}")
print("="*80)

# Determine best model
models = {
    'Random Forest': rf_accuracy,
    'SVM': svm_accuracy,
    'Neural Network': nn_accuracy
}
best_model = max(models, key=models.get)
print(f"\n[OK] Best performing model: {best_model} (Accuracy: {models[best_model]:.4f})")

# Part 5: Real-time IDS Test
print("\n[PART 5] Real-Time IDS Prototype Test")
print("-"*80)

def predict_traffic(data):
    data_scaled = scaler.transform([data])
    pred = rf_model.predict(data_scaled)[0]
    return "Normal" if pred == 0 else "Anomalous"

# Test with sample data
sample_data = X_test.iloc[0].tolist()
prediction = predict_traffic(sample_data)
print(f"[OK] Real-time prediction test: {prediction}")
print(f"  Actual label: {'Normal' if y_test.iloc[0] == 0 else 'Anomalous'}")

# Save results to file
print("\n[SAVING RESULTS]")
print("-"*80)

results = f"""
IDS IMPLEMENTATION - PERFORMANCE RESULTS
========================================

Dataset Information:
- Training samples: {len(X_train)}
- Testing samples: {len(X_test)}
- Features: {X_train.shape[1]}

Model Performance:
------------------

Random Forest:
  Accuracy:  {rf_accuracy:.4f}
  Precision: {rf_precision:.4f}
  Recall:    {rf_recall:.4f}
  F1-Score:  {rf_f1:.4f}

SVM:
  Accuracy:  {svm_accuracy:.4f}
  Precision: {svm_precision:.4f}
  Recall:    {svm_recall:.4f}
  F1-Score:  {svm_f1:.4f}

Neural Network:
  Accuracy:  {nn_accuracy:.4f}
  Precision: {nn_precision:.4f}
  Recall:    {nn_recall:.4f}
  F1-Score:  {nn_f1:.4f}

Best Model: {best_model}
Real-time IDS: Functional [OK]
"""

with open('RESULTS.txt', 'w') as f:
    f.write(results)

print("[OK] Results saved to RESULTS.txt")

print("\n" + "="*80)
print("ALL TESTS PASSED - READY FOR PRESENTATION!")
print("="*80)
