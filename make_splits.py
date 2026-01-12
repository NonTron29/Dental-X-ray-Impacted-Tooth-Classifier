import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv("labels_labeled.csv")

# Keep only what you need
df = df.dropna(subset=["filepath", "impacted_binary"]).copy()
df["impacted_binary"] = df["impacted_binary"].astype(int)

# 80/10/10 split (stratified)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["impacted_binary"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["impacted_binary"]
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df = pd.concat([train_df, val_df, test_df], ignore_index=True)
df.to_csv("labels_labeled_split.csv", index=False)

print(df["split"].value_counts())
print(df.groupby("split")["impacted_binary"].mean())  # sanity check positive rate
