"""
Quick test script for IDS - Simplified version
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
print("IDS IMPLEMENTATION TEST")
print("="*80)

# Load Data
print("\n[1] Loading Data...")
columns = [f'feature_{i}' for i in range(41)] + ['label']
train_data = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
test_data = pd.read_csv('KDDTest+.txt', header=None, names=columns)
print(f"Training: {train_data.shape}, Testing: {test_data.shape}")

# Encode categorical features (fit on combined to avoid unseen values)
print("\n[2] Preprocessing...")
categorical_cols = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_data[col].astype(str), test_data[col].astype(str)])
    le.fit(combined)
    train_data[col] = le.transform(train_data[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))

# Encode labels
le_label = LabelEncoder()
combined_labels = pd.concat([train_data['label'], test_data['label']])
le_label.fit(combined_labels)
train_data['label'] = le_label.transform(train_data['label'])
test_data['label'] = le_label.transform(test_data['label'])

# Separate and scale
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Preprocessing complete!")

# Train Random Forest
print("\n[3] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
print(f"RF - Acc: {rf_acc:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}")

# Train SVM
print("\n[4] Training SVM (may take 5-10 minutes)...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_rec = recall_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, average='weighted', zero_division=0)
print(f"SVM - Acc: {svm_acc:.4f}, Prec: {svm_prec:.4f}, Rec: {svm_rec:.4f}, F1: {svm_f1:.4f}")

# Train Neural Network
print("\n[5] Training Neural Network...")

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
print(f"NN - Acc: {nn_acc:.4f}, Prec: {nn_prec:.4f}, Rec: {nn_rec:.4f}, F1: {nn_f1:.4f}")

# Results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)
print(f"{'Random Forest':<20} {rf_acc:<12.4f} {rf_prec:<12.4f} {rf_rec:<12.4f} {rf_f1:<12.4f}")
print(f"{'SVM':<20} {svm_acc:<12.4f} {svm_prec:<12.4f} {svm_rec:<12.4f} {svm_f1:<12.4f}")
print(f"{'Neural Network':<20} {nn_acc:<12.4f} {nn_prec:<12.4f} {nn_rec:<12.4f} {nn_f1:<12.4f}")
print("="*80)

# Save results
results = f"""IDS IMPLEMENTATION - RESULTS

Dataset: NSL-KDD
Training samples: {len(X_train)}
Testing samples: {len(X_test)}

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
"""

with open('RESULTS.txt', 'w') as f:
    f.write(results)

print("\nResults saved to RESULTS.txt")
print("READY FOR PRESENTATION!")
