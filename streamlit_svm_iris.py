
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

st.set_page_config(page_title="Iris SVM Playground", layout="wide")

st.title("🌸 Iris SVM Playground")
st.write("Interactively explore Support Vector Machines (SVM) on the classic Iris dataset.")

# -----------------------------
# Load dataset
# -----------------------------
iris = datasets.load_iris()
X_full = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Sidebar controls
st.sidebar.header("Settings")

test_size = st.sidebar.slider("Test size (fraction for test set)", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state (reproducibility)", min_value=0, value=42, step=1)

kernel = st.sidebar.selectbox("Kernel", options=["linear", "rbf", "poly"])
C = st.sidebar.slider("C (Regularization)", 0.01, 100.0, 1.0, 0.01)
gamma_mode = st.sidebar.selectbox("Gamma (RBF/Poly)", options=["scale", "auto", "custom"])
gamma_custom = None
if gamma_mode == "custom":
    gamma_custom = st.sidebar.number_input("Gamma value", value=0.1, min_value=0.0001, step=0.01, format="%.4f")

degree = 3
if kernel == "poly":
    degree = st.sidebar.slider("Polynomial degree", 2, 8, 3, 1)

# Feature selection for 2D visualization
st.sidebar.markdown("---")
st.sidebar.subheader("2D Decision Boundary")
f1 = st.sidebar.selectbox("X-axis feature", options=feature_names, index=0)
f2 = st.sidebar.selectbox("Y-axis feature", options=feature_names, index=1)
if f1 == f2:
    st.sidebar.warning("Pick two different features for the 2D plot.")

# Map feature names to indices
f1_idx = feature_names.index(f1)
f2_idx = feature_names.index(f2)

# Prepare 2D feature matrix for visualization and 2D training
X_2d = X_full[:, [f1_idx, f2_idx]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=test_size, stratify=y, random_state=random_state
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build SVM kwargs
svm_kwargs = dict(kernel=kernel, C=C, probability=False, random_state=random_state if kernel == "linear" else None)
if kernel in ("rbf", "poly"):
    if gamma_mode in ("scale", "auto"):
        svm_kwargs["gamma"] = gamma_mode
    else:
        svm_kwargs["gamma"] = float(gamma_custom) if gamma_custom is not None else "scale"
if kernel == "poly":
    svm_kwargs["degree"] = degree

# Train
model = SVC(**svm_kwargs)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# Layout
left, right = st.columns([1, 1])

with left:
    st.subheader("Metrics")
    st.metric(label="Test Accuracy", value=f"{acc:.3f}")
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"true_{n}" for n in target_names], columns=[f"pred_{n}" for n in target_names])
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Classification Report")
    # Convert classification report to DataFrame
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=3), use_container_width=True)

st.markdown("---")
st.subheader("Decision Boundary (2D)")

# Decision boundary plot (2D)
# Only uses the two selected features (already scaled)
# Create meshgrid
x_min, x_max = X_train_scaled[:, 0].min() - 1.0, X_train_scaled[:, 0].max() + 1.0
y_min, y_max = X_train_scaled[:, 1].min() - 1.0, X_train_scaled[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict over grid
Z = model.predict(grid).reshape(xx.shape)

# Plot with matplotlib (single plot, no explicit colors)
fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, Z, alpha=0.25)
# Scatter training points
for i, name in enumerate(target_names):
    idx = (y_train == i)
    ax.scatter(X_train_scaled[idx, 0], X_train_scaled[idx, 1], label=f"train {name}", alpha=0.9, s=30)
# Scatter test points (larger markers with edge)
for i, name in enumerate(target_names):
    idx = (y_test == i)
    ax.scatter(X_test_scaled[idx, 0], X_test_scaled[idx, 1], marker='x', label=f"test {name}", s=40)

ax.set_xlabel(f"{f1} (scaled)")
ax.set_ylabel(f"{f2} (scaled)")
ax.set_title(f"SVM decision regions — kernel={kernel}, C={C}")
ax.legend(loc="best")
st.pyplot(fig)

# Show current parameters
st.markdown("### Current Model Settings")
st.json({
    "kernel": kernel,
    "C": C,
    "gamma": svm_kwargs.get("gamma", None),
    "degree": svm_kwargs.get("degree", None),
    "test_size": test_size,
    "random_state": random_state,
    "features": [f1, f2],
})
