## Détection automatique de désinformation dans les titres de presse
### Pipeline NLP complet — TF-IDF · TensorFlow · FastAPI

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green?logo=fastapi)


## Contexte

Un organisme de vérification des faits (fact-checking) publie chaque semaine des rapports sur la fiabilité des articles de presse en ligne. Il reçoit des milliers de titres d'articles issus de sources variées — certains sont des titres d'articles journalistiques vérifiés, d'autres sont des titres de contenus identifiés comme trompeurs ou sensationnalistes.

Automatiser un premier niveau de triage : classer chaque titre entrant comme **fiable** ou **trompeur**, avant qu'un analyste humain ne prenne la décision finale. Ce pré-filtrage doit diviser par trois la charge de travail manuelle.
---

## Dataset

**Nom :** Fake News Detection Dataset  
**Source :** Kaggle  
**URL :** https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news  
**Fichier principal :** `fake_or_real_news.csv`

**Structure du fichier :**

| Colonne | Type | Description |
|---|---|---|
| `title` | str | Titre de l'article |
| `text` | str | Corps de l'article (non utilisé) |
| `label` | str | `REAL` ou `FAKE` |
  
Le corpus contient environ 6 300 articles. 

---

## Structure

```
fake-news-detection-nlp/
├── notebook/
│   └── fake_news.ipynb      # Notebook principal 
├── api/
│   └── main.py                  # Application FastAPI
├── front/
│   ├── index.html 
│   ├── style.css 
│   └── script.js
├── models/
│   ├── embedding_model.keras         # Meilleur modèle LTSM
│   ├── tfidf_model.keras         # Meilleur modèle TF-IDF
│   └── tfidf_vectorizer.joblib           # Vectoriseur TF-IDF
├── src/     
│   ├── data_cleaning.py
│   └── utils.py 
├── data/
│   ├── fake_or_real_news.csv   # Données brutes
│   └── titles_clean.csv  # Données nettoyées
│
├──  .dockerignore
├──  Dockerfile.yml
├──  docker-compose.yml
└── requirements.txt
```

## Technologies utilisées


## Exécution

1. **Lancer le projet avec Docker Compose :**

```bash
docker compose up -d
```
2. **Accéder à l’API :**

  URL locale : http://localhost:8000

  Documentation Swagger : http://localhost:8000/docs

3. **Interface utilisateur :**

Une fois le conteneur en fonctionnement, ouvrir front/index.html pour tester la détection via le navigateur.