import pandas as pd

df = pd.read_csv("sms_spam_test.csv")
def clean_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s
df["text"] = df["text"].apply(clean_text)
df = df.dropna(subset=["text"])
df.to_csv("sms_spam_test.csv", index=False)
print("已清理 sms_spam_test.csv")
