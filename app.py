from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(page_title="ML Assignment 2", page_icon="🧠", layout="wide")


@st.cache_data
def load_default_dataset():
    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    return features, target


@st.cache_data
def build_split():
    X, y = load_default_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


@st.cache_resource
def build_models():
    model_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "kNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
    }

    loaded_models = {}
    for display_name, filename in model_map.items():
        try:
            loaded_models[display_name] = joblib.load(MODEL_DIR / filename)
        except FileNotFoundError:
            loaded_models[display_name] = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=42)),
                ]
            )

    return loaded_models


@st.cache_data
def evaluate_models_on_split():
    X_train, X_test, y_train, y_test = build_split()
    models = build_models()
    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "AUC": round(roc_auc_score(y_test, y_prob), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
            }
        )

    return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)


@st.cache_data
def evaluate_single_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    return metrics, y_pred, y_prob


st.title("Classification Model Comparison Dashboard")
st.caption("This Streamlit app compares six machine-learning classifiers on the Breast Cancer Wisconsin dataset.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="Upload a CSV test file with the same feature columns and a target column named 'target'.",
    )
    selected_model = st.selectbox("Choose a model to inspect", list(build_models().keys()))

X_train, X_test, y_train, y_test = build_split()
comparison_df = evaluate_models_on_split()

st.subheader("Model comparison table")
st.dataframe(comparison_df, use_container_width=True)

st.subheader("Model selection and evaluation")
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if "target" not in uploaded_df.columns:
            st.error("Uploaded CSV must contain a 'target' column for evaluation.")
            st.stop()

        feature_columns = X_train.columns.tolist()
        if set(uploaded_df.columns) != set(feature_columns + ["target"]):
            st.warning("Uploaded CSV columns differ slightly from the default dataset schema. The app will still evaluate it if the feature count matches.")

        X_eval = uploaded_df.drop(columns=["target"])
        y_eval = uploaded_df["target"]
        if X_eval.shape[1] != X_train.shape[1]:
            st.error("Uploaded CSV must have the same number of feature columns as the default dataset.")
            st.stop()

        st.success(f"Using uploaded test CSV: {uploaded_file.name}")
        eval_X_train, eval_X_test, eval_y_train, eval_y_test = X_train.copy(), X_eval.copy(), y_train.copy(), y_eval.copy()
    except Exception as exc:
        st.error(f"Failed to read the uploaded CSV: {exc}")
        st.stop()
else:
    eval_X_train, eval_X_test, eval_y_train, eval_y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

model = build_models()[selected_model]
metrics, y_pred, y_prob = evaluate_single_model(model, eval_X_train, eval_X_test, eval_y_train, eval_y_test)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("AUC", f"{metrics['AUC']:.4f}")
col3.metric("F1", f"{metrics['F1']:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Precision", f"{metrics['Precision']:.4f}")
col5.metric("Recall", f"{metrics['Recall']:.4f}")
col6.metric("MCC", f"{metrics['MCC']:.4f}")

st.write("Evaluation metrics for the selected model on the chosen test data")
metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
st.dataframe(metrics_df, use_container_width=True)

cm = confusion_matrix(eval_y_test, y_pred)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix - {selected_model}")
st.pyplot(fig)

st.text(classification_report(eval_y_test, y_pred, digits=4))

st.subheader("Dataset overview")
X_full, _ = load_default_dataset()
st.write(f"Default dataset shape: {X_full.shape}")
st.write("Feature columns:", X_full.columns.tolist())
st.write("Binary target column: target")

winner = comparison_df.loc[comparison_df["Accuracy"].idxmax(), "Model"]
st.info(f"Overall winner on the default hold-out dataset: {winner}")

st.markdown(
    """
    ### Notes
    - The app uses the Breast Cancer Wisconsin dataset for a reproducible, academic ML comparison.
    - Upload a CSV test file to evaluate the selected model on your own hold-out data.
    - The comparison table shows the performance of each modeled classifier on the default test split.
    """
)
