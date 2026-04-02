import pandas as pd
import csv

def load_titles(filepath:str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    data = data[["title", "label"]] 
    data = data.rename(columns={"title": "text"})
    data["label"] = data["label"].map({"REAL": 1, "FAKE": 0})
    data = data.dropna(subset=["text"])

    number_title_by_class = data.groupby("label").count()
    print(f"Nombre de titres par classes : {number_title_by_class}")

    data.to_csv("./data/titles_clean.csv", index=False, encoding="utf-8")

    return data

if __name__ == "__main__":
    FILE_PATH = "./data/fake_or_real_news.csv"
    data = load_titles(FILE_PATH)
    print(data.head(5))