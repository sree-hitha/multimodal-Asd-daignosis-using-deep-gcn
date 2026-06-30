# gcn_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
df = pd.read_csv("asd_ready.csv")

# Separate features and labels
X = df.drop("result", axis=1)
y = df["result"]

# -----------------------------
# 🔥 STEP 2: CLEAN DATA (VERY IMPORTANT)
# -----------------------------

# Convert all to numeric (handles '?' etc.)
X = X.apply(pd.to_numeric, errors='coerce')

# Replace NaN with mean
X = X.fillna(X.mean())

# If still any NaN → replace with 0
X = X.fillna(0)

# Convert to numpy
X = X.values
y = y.values

# -----------------------------
# STEP 3: Normalize
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Extra safety (removes any inf/NaN)
X = np.nan_to_num(X)

# -----------------------------
# STEP 4: Create Graph
# -----------------------------
similarity_matrix = cosine_similarity(X)

threshold = 0.7
edge_index = []

for i in range(len(X)):
    for j in range(len(X)):
        if i != j and similarity_matrix[i][j] > threshold:
            edge_index.append([i, j])

# 🔥 Handle case when no edges found
if len(edge_index) == 0:
    print("⚠️ No edges found, lowering threshold...")
    threshold = 0.5
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j and similarity_matrix[i][j] > threshold:
                edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# -----------------------------
# STEP 5: Convert to tensors
# -----------------------------
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# -----------------------------
# STEP 6: Define Model
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

# -----------------------------
# STEP 7: Train Model
# -----------------------------
model = GCN(input_dim=x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training started...\n")

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.nll_loss(out, y)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

# -----------------------------
# STEP 8: Save Model
# -----------------------------
torch.save(model.state_dict(), "model.pth")
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("\n✅ Model Saved Successfully!")