import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import os
from sklearn.model_selection import train_test_split
import numpy as np

# 載入模型
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("svm_best_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("logreg_best_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

# 載入資料
df = pd.read_csv("sms_spam_clean.csv")
df = df[df["text_clean"].notnull() & (df["text_clean"].astype(str).str.strip() != "") & (df["text_clean"].astype(str).str.lower() != "nan")]
X = df["text_clean"]
y = df["label"]

ham = df[df["label"] == 0]
spam = df[df["label"] == 1]
test_ham = ham.sample(n=min(120, len(ham)), random_state=42)
test_spam = spam.sample(n=min(60, len(spam)), random_state=42)
test_set = pd.concat([test_ham, test_spam])
train_set = df.drop(test_set.index)
X_train = train_set["text_clean"]
y_train = train_set["label"]
X_test = test_set["text_clean"]
y_test = test_set["label"]
X_test_vec = vectorizer.transform(X_test)
y_test_bin = y_test

os.makedirs("reports/visualizations", exist_ok=True)

# 類別分布
plt.figure(figsize=(5,4))
sns.countplot(x=df["label"])
plt.title("Class Distribution")
plt.savefig("reports/visualizations/class_distribution.png")
plt.close()

# SVM 預測
svm_pred = svm_model.predict(X_test_vec)
if hasattr(svm_model, "predict_proba"):
    svm_proba = svm_model.predict_proba(X_test_vec)[:,1]
else:
    try:
        svm_proba = svm_model.decision_function(X_test_vec)
    except:
        svm_proba = svm_pred

# 邏輯回歸預測
logreg_pred = logreg_model.predict(X_test_vec)
logreg_proba = logreg_model.predict_proba(X_test_vec)[:,1]

# 混淆矩陣
for name, pred in zip(["SVM", "LogisticRegression"], [svm_pred, logreg_pred]):
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham","spam"], yticklabels=["ham","spam"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"reports/visualizations/confusion_matrix_{name}.png")
    plt.close()

# ROC 曲線
for name, proba in zip(["SVM", "LogisticRegression"], [svm_proba, logreg_proba]):
    fpr, tpr, _ = roc_curve(y_test_bin, proba)
    auc = roc_auc_score(y_test_bin, proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"reports/visualizations/roc_{name}.png")
    plt.close()

# PR 曲線
for name, proba in zip(["SVM", "LogisticRegression"], [svm_proba, logreg_proba]):
    precision, recall, _ = precision_recall_curve(y_test_bin, proba)
    ap = average_precision_score(y_test_bin, proba)
    plt.figure(figsize=(5,4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"reports/visualizations/pr_{name}.png")
    plt.close()

print("所有訓練結果圖表已產生於 reports/visualizations/")