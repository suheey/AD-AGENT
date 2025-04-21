import pandas as pd

df = pd.read_csv("data/yahoo_sub_5.csv")

df = df.sort_values(by="timestamp").reset_index(drop=True)

split_index = int(len(df) * 0.7)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

train_df.to_csv("data/yahoo_train.csv", index=False)
test_df.to_csv("data/yahoo_test.csv", index=False)

print("Train and test datasets have been split and saved.")
