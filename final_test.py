"""
IDS Implementation Test - Working Version
"""
import pandas as pd
import numpy as np
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
print("IDS IMPLEMENTATION TEST - NSL-KDD Dataset")
print("="*80)

# Load Data
print("\n[STEP 1] Loading Data...")
train_data = pd.read_csv('KDDTrain+.txt', header=None)
test_data = pd.read_csv('KDDTest+.txt', header=None)
print(f"Training data: {train_data.shape}")
print(f"Testing data: {test_data.shape}")

# The NSL-KDD has 41 features + 1 label + 1 difficulty level
# Columns: duration, protocol_type, service, flag, ... (41 features total), attack_type, difficulty
# We'll use first 41 columns as features and column 41 as label

print("\n[STEP 2] Preprocessing...")

# Extract features and labels
X_train = train_data.iloc[:, :41].copy()
y_train_raw = train_data.iloc[:, 41]
X_test = test_data.iloc[:, :41].copy()
y_test_raw = test_data.iloc[:, 41]

# Identify categorical columns (protocol_type=1, service=2, flag=3)
categorical_indices = [1, 2, 3]

# Encode categorical features
for idx in categorical_indices:
    le = LabelEncoder()
    # Fit on combined data
    combined = pd.concat([X_train.iloc[:, idx].astype(str), X_test.iloc[:, idx].astype(str)])
    le.fit(combined)
    X_train.iloc[:, idx] = le.transform(X_train.iloc[:, idx].astype(str))
    X_test.iloc[:, idx] = le.transform(X_test.iloc[:, idx].astype(str))

# Convert labels to binary (normal vs attack)
y_train = (y_train_raw != 'normal').astype(int)
y_test = (y_test_raw != 'normal').astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Features: {X_train.shape[1]}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Normal/Attack ratio (train): {(y_train==0).sum()}/{(y_train==1).sum()}")

# Train Random Forest
print("\n[STEP 3] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
print(f"Random Forest - Acc: {rf_acc:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}")

# Train SVM
print("\n[STEP 4] Training SVM (this will take 5-10 minutes)...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_rec = recall_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, average='weighted', zero_division=0)
print(f"SVM - Acc: {svm_acc:.4f}, Prec: {svm_prec:.4f}, Rec: {svm_rec:.4f}, F1: {svm_f1:.4f}")

# Train Neural Network
print("\n[STEP 5] Training Neural Network (10 epochs)...")

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
        return self.fc3(x)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

nn_model = IDSNet(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"  Epoch {epoch+1}/10 complete")

nn_model.eval()
with torch.no_grad():
    nn_outputs = nn_model(X_test_tensor)
    _, nn_pred = torch.max(nn_outputs, 1)
    nn_pred = nn_pred.numpy()

nn_acc = accuracy_score(y_test, nn_pred)
nn_prec = precision_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_rec = recall_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_f1 = f1_score(y_test, nn_pred, average='weighted', zero_division=0)
print(f"Neural Network - Acc: {nn_acc:.4f}, Prec: {nn_prec:.4f}, Rec: {nn_rec:.4f}, F1: {nn_f1:.4f}")

# Results Summary
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)
print(f"{'Random Forest':<20} {rf_acc:<12.4f} {rf_prec:<12.4f} {rf_rec:<12.4f} {rf_f1:<12.4f}")
print(f"{'SVM':<20} {svm_acc:<12.4f} {svm_prec:<12.4f} {svm_rec:<12.4f} {svm_f1:<12.4f}")
print(f"{'Neural Network':<20} {nn_acc:<12.4f} {nn_prec:<12.4f} {nn_rec:<12.4f} {nn_f1:<12.4f}")
print("="*80)

best_model = max([('Random Forest', rf_acc), ('SVM', svm_acc), ('Neural Network', nn_acc)], key=lambda x: x[1])
print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")

# Save results
results = f"""IDS IMPLEMENTATION - PERFORMANCE RESULTS
========================================

Dataset: NSL-KDD
Training samples: {len(X_train)}
Testing samples: {len(X_test)}
Features: {X_train.shape[1]}

Model Performance:
------------------

Random Forest:
  Accuracy:  {rf_acc:.4f}
  Precision: {rf_prec:.4f}
  Recall:    {rf_rec:.4f}
  F1-Score:  {rf_f1:.4f}

SVM:
  Accuracy:  {svm_acc:.4f}
  Precision: {svm_prec:.4f}
  Recall:    {svm_rec:.4f}
  F1-Score:  {svm_f1:.4f}

Neural Network:
  Accuracy:  {nn_acc:.4f}
  Precision: {nn_prec:.4f}
  Recall:    {nn_rec:.4f}
  F1-Score:  {nn_f1:.4f}

Best Model: {best_model[0]}
"""

with open('RESULTS.txt', 'w') as f:
    f.write(results)

print("\nResults saved to RESULTS.txt")
print("\n" + "="*80)
print("ALL TESTS COMPLETE - READY FOR PRESENTATION!")
print("="*80)
