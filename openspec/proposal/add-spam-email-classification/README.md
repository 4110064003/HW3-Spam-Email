# 垃圾郵件分類系統

## 專案簡介
本專案以機器學習方法（SVM、邏輯回歸）實現垃圾郵件分類，並自動化資料預處理、模型訓練、評估與報告產生。

## 執行流程

1. 資料預處理
   - 執行 `data_preprocessing.py`，自動下載並清理資料，分割訓練/測試集。

2. 模型訓練與評估
   - 執行 `train_model.py`，自動完成特徵工程、模型訓練、參數優化、交叉驗證與評估。
   - 產生 `model_report.md`，包含主要指標、最佳參數、混淆矩陣。
   - 儲存最佳模型：`svm_best_model.pkl`、`logreg_best_model.pkl`

## 主要檔案說明
- `data_preprocessing.py`：資料下載、清理、分割
- `train_model.py`：特徵工程、模型訓練、評估、報告產生
- `model_report.md`：模型評估報告
- `svm_best_model.pkl`、`logreg_best_model.pkl`：最佳模型

## 依賴套件
- pandas
- scikit-learn
- requests

## 執行方式
```bash
python data_preprocessing.py
python train_model.py
```

## 聯絡方式
如有問題請聯絡專案負責人。
