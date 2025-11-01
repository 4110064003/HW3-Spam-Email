# HW3 — Spam Email / SMS Classification (AIoT)

此專案為碩士課程作業：建立垃圾郵件 / 簡訊分類系統，包含資料清理、特徵向量化、模型訓練與視覺化結果呈現（混淆矩陣、ROC 與 Precision-Recall 曲線），並提供一個簡易 Streamlit 應用以互動測試。

## 專案內容（檔案概覽）
- `sms_spam_clean.csv` — 已清理的資料集（text_clean / label）
- `sms_spam_train.csv`, `sms_spam_test.csv` — 切分後的訓練/測試檔
- `vectorizer.pkl` — TF-IDF 向量化器（pickle）
- `svm_best_model.pkl`, `logreg_best_model.pkl` — 訓練好的 SVM 與 Logistic Regression 模型（pickle）
- `visualize_results.py` — 產生混淆矩陣、ROC、PR 圖並輸出到 `reports/visualizations/`
- `openspec/proposal/add-spam-email-classification/app.py` — Streamlit 應用（可視化 + 即時預測介面）
- `train_model.py`（或 `openspec/.../train_model.py`）— 訓練模型與完成評估的腳本
- `clean_sms_spam_train.py`, `clean_sms_spam_test.py`, `preprocess_emails.py` — 資料清理與前處理工具
- `reports/visualizations/` — 已生成的圖檔（class distribution、confusion matrices、ROC、PR）

## 快速開始（Windows, cmd / Anaconda）
1. 建議啟用您的 conda 環境（或使用系統 Python）：

   ```cmd
   C:\Users\acer\ANACONDA\Scripts\conda.exe activate C:\Users\acer\ANACONDA
   ```

2. 安裝相依套件（可使用 project 根目錄的 `requirements.txt`）：

   ```cmd
   C:\Users\acer\ANACONDA\python.exe -m pip install -r requirements.txt
   ```

3. 產生視覺化結果（非 interactive）：

   ```cmd
   C:\Users\acer\ANACONDA\python.exe visualize_results.py
   ```

   圖檔會輸出至 `reports/visualizations/`。

4. 啟動 Streamlit 應用：

   ```cmd
   C:\Users\acer\ANACONDA\python.exe -m streamlit run openspec/proposal/add-spam-email-classification/app.py
   ```

   然後在瀏覽器查看介面，App 會顯示類別分布、混淆矩陣、ROC/PR 曲線並提供即時預測區塊。

## 注意事項 / 附註
- 標籤格式：程式內會自動處理 `label` 為 `0/1` 或 `"ham"/"spam"` 的格式，轉換為二元 0/1（spam=1）後計算 ROC/PR。
- 若測試集不含正樣本（所有皆為 ham），ROC/PR 會被跳過並顯示警告；這是合理的保護措施。
- 模型與向量器以 pickle 存放（`vectorizer.pkl`、`*_best_model.pkl`）— 若要更新模型，請重新執行 `train_model.py`。
- 若您要在 GitHub Actions 或其他 CI 中使用此專案，請將敏感資料（例如未公開的資料集或大型二進位模型）考慮放在 Release / Git LFS 或私人儲存空間。

## 常見問題（簡短）
- 無法 import `numpy._core` 或類似 ModuleNotFoundError：請重新安裝 numpy
  ```cmd
  C:\Users\acer\ANACONDA\python.exe -m pip install --upgrade --force-reinstall numpy
  ```
- Streamlit 未安裝：
  ```cmd
  C:\Users\acer\ANACONDA\python.exe -m pip install streamlit
  ```

## 我已做的修改
- 已修正並清理 `openspec/proposal/add-spam-email-classification/app.py`（確保縮排正確、穩健地取得模型分數並處理不同標籤格式）。
- 已將整個專案推上至 GitHub: `https://github.com/4110064003/HW3-Spam-Email`。

## 線上示範
您也可以在以下 Streamlit demo 網站上直接試用應用（已部署）：

https://7114064042-hw3.streamlit.app/

---

若要我：
- 幫您自動安裝 `streamlit` 並啟動 App，或
- 將 `requirements.txt` 擴充為固定版本（pinning），或
- 新增一個簡短的 `run.sh` / `run.bat` 自動化啟動流程，
請告訴我您想要的下一步。