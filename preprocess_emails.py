import pandas as pd
import re

# 讀取原始資料
raw = pd.read_csv("sms_spam_train.csv")
# 假設欄位為 text, label
# 清理 text 欄位
raw["text_clean"] = raw["text"].astype(str).str.strip().str.lower()
raw["text_clean"] = raw["text_clean"].apply(lambda x: re.sub(r'[^a-z0-9 ]', '', x))
raw = raw[raw["text_clean"].notnull() & (raw["text_clean"] != "") & (raw["text_clean"] != "nan")]

# 儲存清理後資料
raw.to_csv("sms_spam_clean.csv", index=False)
print("已產生 sms_spam_clean.csv，text_clean 欄位可用於特徵工程")
