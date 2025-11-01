import pandas as pd
import requests
import io

DATA_URL = "https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/raw/master/Chapter03/datasets/sms_spam_no_header.csv"

# 下載資料集
response = requests.get(DATA_URL)
response.raise_for_status()

# 讀取 CSV
csv_data = response.content.decode('utf-8')
df = pd.read_csv(io.StringIO(csv_data), header=None, names=["label", "text"])


# 資料清理：去除空白、標準化標籤、移除空值

label_map = {"ham": 0, "spam": 1}
df["label"] = df["label"].map(label_map)
df["text"] = df["text"].astype(str).str.strip()
# 移除 text 欄位為 nan、空字串或 null 的資料
nan_mask = (df["text"] == "nan") | (df["text"].isnull()) | (df["text"] == "")
df = df[~nan_mask]

	# 再次移除 text 欄位為 nan 或空字串的資料

# 儲存處理後的資料

# 基本文本預處理：移除特殊字元、轉小寫
import re
def clean_text(text):
	text = text.lower()
	text = re.sub(r'[^a-z0-9 ]', '', text)
	return text
df["text"] = df["text"].apply(clean_text)

# 分割資料集
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 儲存處理後的資料
clean_path = "sms_spam_clean.csv"
train_path = "sms_spam_train.csv"
test_path = "sms_spam_test.csv"
df.to_csv(clean_path, index=False)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Cleaned data saved to {clean_path}")
print(f"Train set saved to {train_path}")
print(f"Test set saved to {test_path}")