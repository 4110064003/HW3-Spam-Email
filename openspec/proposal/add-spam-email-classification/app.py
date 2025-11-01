"""
Streamlit app for spam classification. Clean, tested, and robust: handles label formats and missing positive class cases.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import warnings

warnings.filterwarnings("ignore")


def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("svm_best_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("logreg_best_model.pkl", "rb") as f:
        logreg_model = pickle.load(f)
    return vectorizer, svm_model, logreg_model


def get_proba_or_score(model, X):
    """Return a 1-D score/probability for the positive class.
    If predict_proba exists, return column 1. Else try decision_function. Else return predictions (0/1).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # if binary, take positive class
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        # fallback to first column
        return proba.ravel()
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return np.asarray(score).ravel()
    # last resort: predictions
    return np.asarray(model.predict(X)).ravel()


def ensure_binary_labels(y):
    """Convert various label formats to binary 0/1 (spam=1, ham=0).
    Accepts labels like 0/1 or 'ham'/'spam'.
    """
    if y.dtype == object:
        # strings
        y_bin = (y.astype(str).str.lower() == "spam").astype(int)
    else:
        # numeric
        y_bin = (y == 1).astype(int)
    return y_bin


def main():
    st.set_page_config(page_title="Spam Email Classifier", layout="wide")
    st.title("Spam Email Classifier — AIoT HW3")

    vectorizer, svm_model, logreg_model = load_models()

    # Load and clean data
    df = pd.read_csv("sms_spam_clean.csv")
    df = df[df["text_clean"].notnull()]
    df = df[df["text_clean"].astype(str).str.strip() != ""]
    df = df[df["text_clean"].astype(str).str.lower() != "nan"]

    st.sidebar.subheader("Data info")
    st.sidebar.write(df["label"].value_counts())

    X = df["text_clean"]
    y = df["label"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_vec = vectorizer.transform(X_test)
    y_test_bin = ensure_binary_labels(y_test)

    # Class distribution
    st.subheader("Class distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x=df["label"], ax=ax)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    for name, model in [("SVM", svm_model), ("LogisticRegression", logreg_model)]:
        pred = model.predict(X_test_vec)
        # convert pred to binary 0/1 if needed
        pred_bin = (
            pd.Series(pred).astype(int)
            if np.issubdtype(type(pred[0]), np.integer)
            else (pd.Series(pred).astype(str).str.lower() == "spam").astype(int)
        )
        cm = confusion_matrix(y_test_bin, pred_bin)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")
        st.pyplot(fig)

    # ROC and PR curves
    st.subheader("ROC and Precision-Recall")
    for name, model in [("SVM", svm_model), ("LogisticRegression", logreg_model)]:
        proba = get_proba_or_score(model, X_test_vec)
        # ensure proba is numeric
        proba = np.nan_to_num(proba, nan=0.0)

        if len(np.unique(y_test_bin)) < 2:
            st.warning(f"Test set for {name} does not contain both classes. Skipping ROC/PR plots.")
            continue

        # ROC
        fpr, tpr, _ = roc_curve(y_test_bin, proba)
        auc = roc_auc_score(y_test_bin, proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"ROC Curve — {name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        # PR
        precision, recall, _ = precision_recall_curve(y_test_bin, proba)
        ap = average_precision_score(y_test_bin, proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"AP={ap:.3f}")
        ax.set_title(f"Precision-Recall — {name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)

    # Simple prediction UI
    st.header("Quick predict")
    user_input = st.text_area("Enter a message to classify:")
    model_choice = st.selectbox("Model", ["SVM", "LogisticRegression"])
    if st.button("Predict") and user_input.strip():
        txt = str(user_input)
        X_new = vectorizer.transform([txt])
        model = svm_model if model_choice == "SVM" else logreg_model
        proba = get_proba_or_score(model, X_new)
        # if model gives probability-like score, try to normalize
        if proba.size > 0:
            score = float(proba.ravel()[0])
        else:
            score = float(model.predict(X_new)[0])
        # heuristics: if score in [0,1] treat as probability, else apply sigmoid
        if 0.0 <= score <= 1.0:
            spam_pct = score * 100
        else:
            spam_pct = 100 / (1 + np.exp(-score))
        ham_pct = 100 - spam_pct
        st.write(f"Spam probability: {spam_pct:.2f}%")
        st.write(f"Ham probability: {ham_pct:.2f}%")
        st.success("Prediction: " + ("Spam" if spam_pct > ham_pct else "Ham"))


if __name__ == "__main__":
    main()