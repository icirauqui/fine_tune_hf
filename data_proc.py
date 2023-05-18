import pandas as pd

df = pd.read_csv("bitcoin-sentiment-tweets.csv")
print(df.head())
print(df.sentiment.value_counts())
df.sentiment.value_counts().plot(kind='bar');

def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"
 
dataset_data = [
    {
        "instruction": "Detect the sentiment of the tweet.",
        "input": row_dict["tweet"],
        "output": sentiment_score_to_name(row_dict["sentiment"])
    }
    for row_dict in df.to_dict(orient="records")
]
 
(dataset_data[0])

import json
with open("alpaca-bitcoin-sentiment-dataset.json", "w") as f:
   json.dump(dataset_data, f)