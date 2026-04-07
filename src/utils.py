import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords') # python -m spacy download en_core_web_sm
stop_words = set(stopwords.words('english'))
negations = {"not", "no", "never", "neither"} # On exclue ces stopwords car elle influent sur le sens de la phrase
custom_stop_words = stop_words - negations

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

contractions_dict = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "won't": "will not",
        "wouldn't": "would not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "i'm": "i am",
        "you're": "you are",
        "they're": "they are",
        "it's": "it is",
        "there's": "there is",
        "we're": "we are",
        "she's": "she is",
        "he's": "he is"
    }

def clean_title(text:str) -> str:
    text = text.lower() # Mise en minuscules
    text = re.sub(r'@\w+', '', text) # Suppression des URLs et des mentions de type @username
    text = re.sub(r'[^\w\s]', '', text)  # Suppression de la ponctuation
    text = re.sub(r'\b\d+\b', '', text)  # Suppression des nombres isolés
    
    contraction_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions_dict.keys()) + r')\b') # Remplace les contractions
    text = contraction_pattern.sub(lambda x: contractions_dict[x.group()], text)

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in custom_stop_words]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    
    return tokens
    

