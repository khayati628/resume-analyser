from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils import clean_resume

DATA_PATH = Path("data/Resume.csv")
MODEL_DIR = Path("models")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Resume_str", "Category"])
    df = df.drop_duplicates(subset="Resume_str")
    df["cleaned_resume"] = df["Resume_str"].apply(clean_resume)
    return df["cleaned_resume"], df["Category"]


def build_pipeline():
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        sublinear_tf=True,
    )
    model = LogisticRegression(C=10, max_iter=1000)
    return tfidf, model


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. See the Dataset section in README.md.")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    tfidf, model = build_pipeline()
    model.fit(tfidf.fit_transform(X_train), y_train)

    predictions = model.predict(tfidf.transform(X_test))
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions, zero_division=0))

    tfidf, model = build_pipeline()
    model.fit(tfidf.fit_transform(X), y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(tfidf, MODEL_DIR / "tfidf.pkl")
    joblib.dump(model, MODEL_DIR / "classifier.pkl")
    print(f"Saved model files to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
