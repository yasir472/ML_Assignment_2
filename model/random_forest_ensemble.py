from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).with_suffix(".joblib")


def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
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
