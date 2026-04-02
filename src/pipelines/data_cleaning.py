import pandas as pd
import re
import nltk

import spacy
nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords
nltk.download('stopwords') # python -m spacy download en_core_web_sm
stop_words = set(stopwords.words('english'))
negations = {"not", "no", "never", "neither"} # On exclue ces stopwords car elle influent sur le sens de la phrase
custom_stop_words = stop_words - negations

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

    text = nlp(text) # Lemmatisation

    tokens = []
    for token in text:
        if token.lemma_ not in custom_stop_words and len(token.lemma_) >= 2: # Suppression des tokens de longueur inférieure à 2 caractères après lemmatisation
            tokens.append(token.lemma_)
    
    return tokens
    
if __name__ == "__main__":
    data = pd.read_csv("./data/titles_clean.csv")
    print(data)
    data["text_clean"] = data["text"].apply(clean_title)

    print(data[["text", "text_clean"]].head())

    print(data)