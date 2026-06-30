from flask import Flask, render_template, request, redirect, session
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = "secret123"   # 🔐 Required for session

# -----------------------------
# 🔐 LOGIN CREDENTIALS
# -----------------------------
USERNAME = "admin"
PASSWORD = "1234"

# -----------------------------
# ✅ Load scaler
# -----------------------------
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

# -----------------------------
# ✅ Define SAME model
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
# ✅ Load model
# -----------------------------
model = GCN(input_dim=12)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# 🔐 LOGIN PAGE
# -----------------------------
@app.route("/")
def login():
    return render_template("login.html")

# -----------------------------
# 🔐 HANDLE LOGIN
# -----------------------------
@app.route("/login", methods=["POST"])
def handle_login():
    username = request.form["username"]
    password = request.form["password"]

    if username == USERNAME and password == PASSWORD:
        session["user"] = username
        return redirect("/home")
    else:
        return render_template("login.html", error="Invalid Credentials")

# -----------------------------
# 🏠 HOME PAGE (PROTECTED)
# -----------------------------
@app.route("/home")
def home():
    if "user" in session:
        return render_template("home.html")
    return redirect("/")

# -----------------------------
# 🧠 PREDICT PAGE (PROTECTED)
# -----------------------------
@app.route("/predict")
def predict_page():
    if "user" in session:
        return render_template("predict.html")
    return redirect("/")

# -----------------------------
# 🎯 RESULT PAGE (PROTECTED)
# -----------------------------
@app.route("/predict_result", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    try:
        values = []

        # A1-A10
        for i in range(1, 11):
            values.append(int(request.form.get(f"A{i}", 0)))

        # Age & Gender
        age = float(request.form.get("age", 0))
        gender = int(request.form.get("gender", 0))

        values.extend([age, gender])

        # Convert to numpy
        x = np.array(values).reshape(1, -1)

        # Normalize
        x = (x - mean) / scale
        x = np.nan_to_num(x)

        x = torch.tensor(x, dtype=torch.float)

        # Dummy graph
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        # Prediction
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1).item()

        result = "🧠 ASD Detected" if pred == 1 else "✅ No ASD"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

# -----------------------------
# 📊 METRICS PAGE (PROTECTED)
# -----------------------------
@app.route("/metrics")
def metrics():
    if "user" in session:
        return render_template("metrics.html")
    return redirect("/")

# -----------------------------
# 🔓 LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)