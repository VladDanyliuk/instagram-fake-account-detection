import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

print("=" * 60)
print("BASELINE MODELS")
print("=" * 60)

# Load data
df = pd.read_csv('../data/train.csv')

# Features (numeric only)
feature_cols = ['#followers', '#follows', 'nums/length username',
                'fullname words', '#posts', 'description length']

X = df[feature_cols].fillna(0)
y = df['fake']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression
print("\n" + "=" * 60)
print("1. LOGISTIC REGRESSION")
print("=" * 60)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# 2. Random Forest
print("\n" + "=" * 60)
print("2. RANDOM FOREST")
print("=" * 60)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# 3. Gradient Boosting (BASELINE)
print("\n" + "=" * 60)
print("3. GRADIENT BOOSTING (BASELINE)")
print("=" * 60)
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_gb)
print(f"Accuracy: {baseline_acc:.4f}")
print(classification_report(y_test, y_pred_gb))

# Save baseline
with open('../models/gradient_boosting_baseline.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

# ============================================================
# 4. NEURAL NETWORK (PyTorch)
# ============================================================
print("\n" + "=" * 60)
print("4. NEURAL NETWORK (PyTorch)")
print("=" * 60)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# Define the neural network architecture
class FakeAccountNet(nn.Module):
    """
    A feedforward neural network for binary classification of Instagram accounts.
    Architecture: Input -> 64 -> 32 -> 16 -> 1
    Uses BatchNorm and Dropout for regularisation.
    """

    def __init__(self, input_size):
        super(FakeAccountNet, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: Input -> 64 neurons
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 64 -> 32 neurons
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 32 -> 16 neurons
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer: 16 -> 1 (binary classification)
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Prepare PyTorch tensors from scaled data
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialise model, loss function and optimiser
model = FakeAccountNet(input_size=X_train_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Training loop
NUM_EPOCHS = 100
best_acc = 0.0
best_epoch = 0

print(f"\nTraining for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_preds = (test_outputs >= 0.5).float()
            epoch_acc = accuracy_score(y_test.values, test_preds.cpu().numpy())

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Test Acc: {epoch_acc:.4f}")

        # Track best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), '../models/neural_network.pt')

# Final evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    y_pred_nn = (test_outputs >= 0.5).float().cpu().numpy().flatten()
nn_acc = accuracy_score(y_test, y_pred_nn)
print(f"\nNeural Network Final Accuracy: {nn_acc:.4f}")
print(f"Best Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
print(classification_report(y_test, y_pred_nn))

print(f"\nModel architecture:\n{model}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)
print(f"Logistic Regression:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Random Forest:        {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Gradient Boosting:    {baseline_acc:.4f}")
print(f"Neural Network:       {nn_acc:.4f}")
print(f"\nModels saved:")
print(f"  models/gradient_boosting_baseline.pkl")
print(f"  models/neural_network.pt")
print("=" * 60)