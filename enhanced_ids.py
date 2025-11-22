"""
Enhanced IDS Implementation with Improved Accuracy
Includes: Hyperparameter tuning, feature selection, and optimized models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED IDS IMPLEMENTATION - IMPROVED ACCURACY")
print("="*80)

# Load Data
print("\n[STEP 1] Loading Data...")
train_data = pd.read_csv('KDDTrain+.txt', header=None)
test_data = pd.read_csv('KDDTest+.txt', header=None)
print(f"Training: {train_data.shape}, Testing: {test_data.shape}")

# Preprocessing
print("\n[STEP 2] Enhanced Preprocessing...")
X_train = train_data.iloc[:, :41].copy()
y_train_raw = train_data.iloc[:, 41]
X_test = test_data.iloc[:, :41].copy()
y_test_raw = test_data.iloc[:, 41]

# Encode categorical features
categorical_indices = [1, 2, 3]
for idx in categorical_indices:
    le = LabelEncoder()
    combined = pd.concat([X_train.iloc[:, idx].astype(str), X_test.iloc[:, idx].astype(str)])
    le.fit(combined)
    X_train.iloc[:, idx] = le.transform(X_train.iloc[:, idx].astype(str))
    X_test.iloc[:, idx] = le.transform(X_test.iloc[:, idx].astype(str))

# Binary labels
y_train = (y_train_raw != 'normal').astype(int)
y_test = (y_test_raw != 'normal').astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection - Select top 35 features
print("\n[STEP 3] Feature Selection...")
selector = SelectKBest(f_classif, k=35)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
print(f"Selected {X_train_selected.shape[1]} best features from {X_train_scaled.shape[1]}")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Model 1: Optimized Random Forest
print("\n[STEP 4] Training Optimized Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,        # Increased from 100
    max_depth=20,            # Limit depth to prevent overfitting
    min_samples_split=5,     # Require more samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    max_features='sqrt',     # Use sqrt of features
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_selected, y_train)
rf_pred = rf_model.predict(X_test_selected)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
print(f"Optimized RF - Acc: {rf_acc:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}")

# Model 2: Optimized SVM
print("\n[STEP 5] Training Optimized SVM...")
svm_model = SVC(
    kernel='rbf',
    C=10,                    # Increased regularization parameter
    gamma='scale',           # Automatic gamma calculation
    random_state=42
)
svm_model.fit(X_train_selected, y_train)
svm_pred = svm_model.predict(X_test_selected)

svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_rec = recall_score(y_test, svm_pred, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, average='weighted', zero_division=0)
print(f"Optimized SVM - Acc: {svm_acc:.4f}, Prec: {svm_prec:.4f}, Rec: {svm_rec:.4f}, F1: {svm_f1:.4f}")

# Model 3: Gradient Boosting (NEW!)
print("\n[STEP 6] Training Gradient Boosting (NEW MODEL)...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
gb_model.fit(X_train_selected, y_train)
gb_pred = gb_model.predict(X_test_selected)

gb_acc = accuracy_score(y_test, gb_pred)
gb_prec = precision_score(y_test, gb_pred, average='weighted', zero_division=0)
gb_rec = recall_score(y_test, gb_pred, average='weighted', zero_division=0)
gb_f1 = f1_score(y_test, gb_pred, average='weighted', zero_division=0)
print(f"Gradient Boosting - Acc: {gb_acc:.4f}, Prec: {gb_prec:.4f}, Rec: {gb_rec:.4f}, F1: {gb_f1:.4f}")

# Model 4: Enhanced Neural Network
print("\n[STEP 7] Training Enhanced Neural Network...")

class EnhancedIDSNet(nn.Module):
    def __init__(self, input_size):
        super(EnhancedIDSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)      # Increased from 128
        self.bn1 = nn.BatchNorm1d(256)             # Batch normalization
        self.fc2 = nn.Linear(256, 128)             # Increased from 64
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)              # Additional layer
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)             # Reduced dropout

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Increased batch size

nn_model = EnhancedIDSNet(X_train_selected.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

print("Training Enhanced Neural Network (20 epochs)...")
for epoch in range(20):  # Increased epochs
    nn_model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")

nn_model.eval()
with torch.no_grad():
    nn_outputs = nn_model(X_test_tensor)
    _, nn_pred = torch.max(nn_outputs, 1)
    nn_pred = nn_pred.numpy()

nn_acc = accuracy_score(y_test, nn_pred)
nn_prec = precision_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_rec = recall_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_f1 = f1_score(y_test, nn_pred, average='weighted', zero_division=0)
print(f"Enhanced NN - Acc: {nn_acc:.4f}, Prec: {nn_prec:.4f}, Rec: {nn_rec:.4f}, F1: {nn_f1:.4f}")

# Ensemble Model (Voting)
print("\n[STEP 8] Creating Ensemble Model...")
# Combine predictions using majority voting
ensemble_pred = []
for i in range(len(rf_pred)):
    votes = [rf_pred[i], svm_pred[i], gb_pred[i], nn_pred[i]]
    ensemble_pred.append(max(set(votes), key=votes.count))

ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_prec = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
ensemble_rec = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
print(f"Ensemble (Voting) - Acc: {ensemble_acc:.4f}, Prec: {ensemble_prec:.4f}, Rec: {ensemble_rec:.4f}, F1: {ensemble_f1:.4f}")

# Results Summary
print("\n" + "="*80)
print("ENHANCED RESULTS COMPARISON")
print("="*80)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)
print(f"{'Optimized RF':<25} {rf_acc:<12.4f} {rf_prec:<12.4f} {rf_rec:<12.4f} {rf_f1:<12.4f}")
print(f"{'Optimized SVM':<25} {svm_acc:<12.4f} {svm_prec:<12.4f} {svm_rec:<12.4f} {svm_f1:<12.4f}")
print(f"{'Gradient Boosting (NEW)':<25} {gb_acc:<12.4f} {gb_prec:<12.4f} {gb_rec:<12.4f} {gb_f1:<12.4f}")
print(f"{'Enhanced Neural Net':<25} {nn_acc:<12.4f} {nn_prec:<12.4f} {nn_rec:<12.4f} {nn_f1:<12.4f}")
print(f"{'Ensemble (Voting)':<25} {ensemble_acc:<12.4f} {ensemble_prec:<12.4f} {ensemble_rec:<12.4f} {ensemble_f1:<12.4f}")
print("="*80)

# Find best model
models_dict = {
    'Optimized RF': rf_acc,
    'Optimized SVM': svm_acc,
    'Gradient Boosting': gb_acc,
    'Enhanced Neural Net': nn_acc,
    'Ensemble': ensemble_acc
}
best_model = max(models_dict, key=models_dict.get)
print(f"\nBest Model: {best_model} (Accuracy: {models_dict[best_model]:.4f})")

# Save enhanced results
results = f"""ENHANCED IDS IMPLEMENTATION - IMPROVED RESULTS
==============================================

Optimizations Applied:
- Feature Selection (35 best features)
- Hyperparameter Tuning
- Gradient Boosting added
- Enhanced Neural Network (4 layers, batch norm)
- Ensemble Voting

Model Performance:
------------------

Optimized Random Forest:
  Accuracy:  {rf_acc:.4f}
  Precision: {rf_prec:.4f}
  Recall:    {rf_rec:.4f}
  F1-Score:  {rf_f1:.4f}

Optimized SVM:
  Accuracy:  {svm_acc:.4f}
  Precision: {svm_prec:.4f}
  Recall:    {svm_rec:.4f}
  F1-Score:  {svm_f1:.4f}

Gradient Boosting (NEW):
  Accuracy:  {gb_acc:.4f}
  Precision: {gb_prec:.4f}
  Recall:    {gb_rec:.4f}
  F1-Score:  {gb_f1:.4f}

Enhanced Neural Network:
  Accuracy:  {nn_acc:.4f}
  Precision: {nn_prec:.4f}
  Recall:    {nn_rec:.4f}
  F1-Score:  {nn_f1:.4f}

Ensemble (Voting):
  Accuracy:  {ensemble_acc:.4f}
  Precision: {ensemble_prec:.4f}
  Recall:    {ensemble_rec:.4f}
  F1-Score:  {ensemble_f1:.4f}

Best Model: {best_model}
"""

with open('ENHANCED_RESULTS.txt', 'w') as f:
    f.write(results)

print("\nEnhanced results saved to ENHANCED_RESULTS.txt")
print("\n" + "="*80)
print("ACCURACY IMPROVEMENT COMPLETE!")
print("="*80)
