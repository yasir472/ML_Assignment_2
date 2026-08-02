from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path(__file__).with_suffix(".joblib")


def build_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved: {MODEL_PATH.name}")
