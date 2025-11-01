# 依照官方流程，使用 sms_spam_clean.csv 的 text_clean 欄位
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import pickle
import numpy as np

df = pd.read_csv("sms_spam_clean.csv")
df = df[df["text_clean"].notnull() & (df["text_clean"].astype(str).str.strip() != "") & (df["text_clean"].astype(str).str.lower() != "nan")]
X = df["text_clean"]
y = df["label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# TF-IDF 特徵工程
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVM
svm_params = {"C": [0.1, 1, 2, 10], "kernel": ["linear", "rbf"], "class_weight": ["balanced"]}
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring="f1", n_jobs=-1)
svm_grid.fit(X_train_vec, y_train)
svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(X_test_vec)
svm_cv_score = cross_val_score(svm_best, X_train_vec, y_train, cv=5, scoring="f1")
with open("svm_best_model.pkl", "wb") as f:
	pickle.dump(svm_best, f)

# 邏輯回歸
logreg_params = {"C": [0.1, 1, 2, 10], "penalty": ["l2"], "solver": ["lbfgs"], "class_weight": ["balanced"]}
logreg_grid = GridSearchCV(LogisticRegression(max_iter=200, random_state=42), logreg_params, cv=5, scoring="f1", n_jobs=-1)
logreg_grid.fit(X_train_vec, y_train)
logreg_best = logreg_grid.best_estimator_
logreg_pred = logreg_best.predict(X_test_vec)
logreg_cv_score = cross_val_score(logreg_best, X_train_vec, y_train, cv=5, scoring="f1")
with open("logreg_best_model.pkl", "wb") as f:
	pickle.dump(logreg_best, f)

# 儲存 vectorizer
with open("vectorizer.pkl", "wb") as f:
	pickle.dump(vectorizer, f)

# 評估指標
svm_report = classification_report(y_test, svm_pred, target_names=["ham", "spam"], output_dict=True)
logreg_report = classification_report(y_test, logreg_pred, target_names=["ham", "spam"], output_dict=True)
svm_cm = confusion_matrix(y_test, svm_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)
with open("model_report.md", "w", encoding="utf-8") as f:
	f.write("# 模型評估報告\n\n")
	f.write("## SVM\n")
	f.write(f"最佳參數: {svm_grid.best_params_}\n")
	f.write(f"交叉驗證 F1 分數: {svm_cv_score.mean():.4f}\n\n")
	f.write("### 評估指標\n")
	for label in ["ham", "spam"]:
		f.write(f"- {label}: ")
		f.write(f"Precision: {svm_report[label]['precision']:.3f}, ")
		f.write(f"Recall: {svm_report[label]['recall']:.3f}, ")
		f.write(f"F1: {svm_report[label]['f1-score']:.3f}\n")
	f.write(f"- Accuracy: {svm_report['accuracy']:.3f}\n")
	f.write("### 混淆矩陣\n")
	f.write(str(svm_cm) + "\n\n")
	f.write("## 邏輯回歸\n")
	f.write(f"最佳參數: {logreg_grid.best_params_}\n")
	f.write(f"交叉驗證 F1 分數: {logreg_cv_score.mean():.4f}\n\n")
	f.write("### 評估指標\n")
	for label in ["ham", "spam"]:
		f.write(f"- {label}: ")
		f.write(f"Precision: {logreg_report[label]['precision']:.3f}, ")
		f.write(f"Recall: {logreg_report[label]['recall']:.3f}, ")
		f.write(f"F1: {logreg_report[label]['f1-score']:.3f}\n")
	f.write(f"- Accuracy: {logreg_report['accuracy']:.3f}\n")
	f.write("### 混淆矩陣\n")
	f.write(str(logreg_cm) + "\n")
print("模型評估報告已產生：model_report.md")
# 依照官方流程，使用 sms_spam_clean.csv 的 text_clean 欄位
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle

df = pd.read_csv("sms_spam_clean.csv")
df = df[df["text_clean"].notnull() & (df["text_clean"].astype(str).str.strip() != "") & (df["text_clean"].astype(str).str.lower() != "nan")]

# 切分資料
from sklearn.model_selection import train_test_split
X = df["text_clean"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF 特徵工程
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVM
svm_params = {"C": [0.1, 1, 2, 10], "kernel": ["linear", "rbf"], "class_weight": ["balanced"]}
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring="f1", n_jobs=-1)
svm_grid.fit(X_train_vec, y_train)
svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(X_test_vec)
svm_cv_score = cross_val_score(svm_best, X_train_vec, y_train, cv=5, scoring="f1")
with open("svm_best_model.pkl", "wb") as f:
	pickle.dump(svm_best, f)

# 邏輯回歸
logreg_params = {"C": [0.1, 1, 2, 10], "penalty": ["l2"], "solver": ["lbfgs"], "class_weight": ["balanced"]}
logreg_grid = GridSearchCV(LogisticRegression(max_iter=200, random_state=42), logreg_params, cv=5, scoring="f1", n_jobs=-1)
logreg_grid.fit(X_train_vec, y_train)
logreg_best = logreg_grid.best_estimator_
logreg_pred = logreg_best.predict(X_test_vec)
logreg_cv_score = cross_val_score(logreg_best, X_train_vec, y_train, cv=5, scoring="f1")
with open("logreg_best_model.pkl", "wb") as f:
	pickle.dump(logreg_best, f)

# 儲存 vectorizer
with open("vectorizer.pkl", "wb") as f:
	pickle.dump(vectorizer, f)

# 評估指標
svm_report = classification_report(y_test, svm_pred, target_names=["ham", "spam"], output_dict=True)
logreg_report = classification_report(y_test, logreg_pred, target_names=["ham", "spam"], output_dict=True)
svm_cm = confusion_matrix(y_test, svm_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)
with open("model_report.md", "w", encoding="utf-8") as f:
	f.write("# 模型評估報告\n\n")
	f.write("## SVM\n")
	f.write(f"最佳參數: {svm_grid.best_params_}\n")
	f.write(f"交叉驗證 F1 分數: {svm_cv_score.mean():.4f}\n\n")
	f.write("### 評估指標\n")
	for label in ["ham", "spam"]:
		f.write(f"- {label}: ")
	f.write(f"Precision: {svm_report[label]['precision']:.3f}, ")
	f.write(f"Recall: {svm_report[label]['recall']:.3f}, ")
	f.write(f"F1: {svm_report[label]['f1-score']:.3f}\n")
	f.write(f"- Accuracy: {svm_report['accuracy']:.3f}\n")
	f.write("### 混淆矩陣\n")
	f.write(str(svm_cm) + "\n\n")
	f.write("## 邏輯回歸\n")
	f.write(f"最佳參數: {logreg_grid.best_params_}\n")
	f.write(f"交叉驗證 F1 分數: {logreg_cv_score.mean():.4f}\n\n")
	f.write("### 評估指標\n")
	for label in ["ham", "spam"]:
		f.write(f"- {label}: ")
	f.write(f"Precision: {logreg_report[label]['precision']:.3f}, ")
	f.write(f"Recall: {logreg_report[label]['recall']:.3f}, ")
	f.write(f"F1: {logreg_report[label]['f1-score']:.3f}\n")
	f.write(f"- Accuracy: {logreg_report['accuracy']:.3f}\n")
	f.write("### 混淆矩陣\n")
	f.write(str(logreg_cm) + "\n")
print("模型評估報告已產生：model_report.md")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle

# 讀取資料
train_df = pd.read_csv("sms_spam_train.csv")
test_df = pd.read_csv("sms_spam_test.csv")
# 偵錯：清理前異常值統計
for name, df in zip(["train", "test"], [train_df, test_df]):
    print(f"{name} 清理前 text 欄位 nan 數量:", df["text"].isnull().sum())
    print(f"{name} 清理前 text 欄位空字串數量:", (df["text"].astype(str).str.strip() == "").sum())
    print(f"{name} 清理前 text 欄位 'nan' 字串數量:", (df["text"].astype(str).str.lower() == "nan").sum())
# 最嚴格清理 text 欄位
def clean_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s
for df in [train_df, test_df]:
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
# 偵錯：清理後異常值統計
for name, df in zip(["train", "test"], [train_df, test_df]):
    print(f"{name} 清理後 text 欄位 nan 數量:", df["text"].isnull().sum())
    print(f"{name} 清理後 text 欄位空字串數量:", (df["text"].astype(str).str.strip() == "").sum())
    print(f"{name} 清理後 text 欄位 'nan' 字串數量:", (df["text"].astype(str).str.lower() == "nan").sum())

# 特徵工程前，徹底清理 text 欄位
for df in [train_df, test_df]:
	df = df[df["text"].notnull() & (df["text"].astype(str).str.strip() != "") & (df["text"].astype(str).str.lower() != "nan")]

vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
y_train = train_df["label"]
y_test = test_df["label"]


# SVM 參數優化
svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring="f1", n_jobs=-1)
svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(X_test)

# SVM 交叉驗證分數
svm_cv_score = cross_val_score(svm_best, X_train, y_train, cv=5, scoring="f1")

# 儲存最佳 SVM 模型
with open("svm_best_model.pkl", "wb") as f:
	pickle.dump(svm_best, f)

# 邏輯回歸參數優化
logreg_params = {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
logreg_grid = GridSearchCV(LogisticRegression(max_iter=200, random_state=42), logreg_params, cv=5, scoring="f1", n_jobs=-1)
logreg_grid.fit(X_train, y_train)
logreg_best = logreg_grid.best_estimator_
logreg_pred = logreg_best.predict(X_test)

# 邏輯回歸交叉驗證分數
logreg_cv_score = cross_val_score(logreg_best, X_train, y_train, cv=5, scoring="f1")

# 儲存最佳邏輯回歸模型
with open("logreg_best_model.pkl", "wb") as f:
	pickle.dump(logreg_best, f)

# 評估指標

svm_report = classification_report(y_test, svm_pred, target_names=["ham", "spam"], output_dict=True)
logreg_report = classification_report(y_test, logreg_pred, target_names=["ham", "spam"], output_dict=True)
svm_cm = confusion_matrix(y_test, svm_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)

with open("model_report.md", "w", encoding="utf-8") as f:
	f.write("# 模型評估報告\n\n")
	f.write("## SVM\n")
	f.write(f"最佳參數: {svm_grid.best_params_}\n")
	f.write(f"交叉驗證 F1 分數: {svm_cv_score.mean():.4f}\n\n")
	f.write("### 評估指標\n")
	for label in ["ham", "spam"]:
		f.write(f"- {label}: ")
		f.write(f"Precision: {svm_report[label]['precision']:.3f}, ")
		f.write(f"Recall: {svm_report[label]['recall']:.3f}, ")
		f.write(f"F1: {svm_report[label]['f1-score']:.3f}\n")
	f.write(f"- Accuracy: {svm_report['accuracy']:.3f}\n")
	f.write("### 混淆矩陣\n")
	f.write(str(svm_cm) + "\n\n")

	f.write("## 邏輯回歸\n")
	f.write(f"最佳參數: {logreg_grid.best_params_}\n")
	f.write(f"交叉驗證 F1 分數: {logreg_cv_score.mean():.4f}\n\n")
	f.write("### 評估指標\n")
	for label in ["ham", "spam"]:
		f.write(f"- {label}: ")
		f.write(f"Precision: {logreg_report[label]['precision']:.3f}, ")
		f.write(f"Recall: {logreg_report[label]['recall']:.3f}, ")
		f.write(f"F1: {logreg_report[label]['f1-score']:.3f}\n")
	f.write(f"- Accuracy: {logreg_report['accuracy']:.3f}\n")
	f.write("### 混淆矩陣\n")
	f.write(str(logreg_cm) + "\n")

print("模型評估報告已產生：model_report.md")