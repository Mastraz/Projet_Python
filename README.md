# 🎬💙 Moveetic ! Movies Recommandation 
*Master 1, ISEN, January to February 2026, Teacher: Cyril Barrelet*

Analyzing movie synopses based on user input,
Using : NLP (BERT), Cosine Similarity and an API interface (FastAPI).


## 📖 1) Project Summary
Mooveetic is an intelligent movie recommendation engine that uses Natural Language Processing (NLP) and Deep Learning. Our code can understand the meaning of movie plots to suggest films with similar themes and narratives.

### Key Features:
- Data Cleaning: preprocessing of the TMDB dataset, including JSON parsing and handling missing values.
- Vectorization : transformation of movie synopses into dimensional vectors (Embeddings) using the BERT model.
-  Recommendation : High-speed similarity calculation using Cosine Similarity to find the closest matches in the database.
-  API: A clean and documented FastAPI interface allowing real-time movie analysis and recommendation display.


### General System Schema:

 <img width="776" height="236" alt="Capture d’écran 2026-02-04 091744" src="https://github.com/user-attachments/assets/8419e908-0e69-4933-a08a-ee3f05b51421" />



## 💽 2) Setup Tutorial
To ensure the project runs smoothly, please follow the steps below.

### Option 1 : Conda

```Bash
# 1. Clone the project
git clone https://github.com/Mastraz/Moveetic.git
cd Moveetic

# 2. Create the environment from the yml file
conda env create -f environment.yml

# 3. Activate the environment
conda activate Moveetic_env
```

### Option 2 : Pip
Not using Conda ? Make sure you have Python 3.12 installed.

```Bash
# 1. Clone the project
git clone https://github.com/Mastraz/Moveetic.git
cd Moveetic

# 2. Download dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
## 🚀 Using
The project is fully automated. At the first launch, it will clean the data and generate the mathematical vectors (Embeddings).

Starting the server 
Launch the API with Uvicorn :

```Bash
python main.py
```
The server will be accessible at : http://127.0.0.1:8000
---
# API test

The easiest way is to use the built-in Swagger interface : 

Open your browser on : http://127.0.0.1:8000/docs.

Unfold the GET/analyze route.

Click on "Try it out".

Enter a title and a text (ex: a heist movie in space with dogs).

Click on Execute to see the 3 best recommendations from the TMDB database.





## 🧩 2) Features

## A) Data Cleaning
### Data Cleaning
Dataset TMDB link : TMDB 5000 Movie Dataset

This feature is a crucial first step. It transforms the raw TMDB dataset into a structured format that is optimized for BERT and the API. 

### Main functionalities:
- Automation: load, clean and save data at once, using MovieCleaner (OO) class.
- Parsing JSON : transforms complex JSON strings into readable text lists.
e.g: '[{"id": 28, "name": "Action"}]' -> 'Action, Adventure'
- Missing values : deletes any row missing key information to prevent future failures/
- Export production : generate the file "movies_cleaned.csv" 

### Rows selection strategy: 
Reduce the size of the dataset to increase computing power, from 19 rows to 6.

Explanation for each column kept:

| Colonne | Pertinence pour le projet |
| :--- | :--- |
| **id** | **Technical Key:** Essential for linking results to the TMDB API. |
| **title** | **User Interface:** To display the results in a clear way. |
| **overview** | **NLP Core:** Contains the synopses used by our BERT model. |
| **genres** | **Filtering:** Used to create category-based filters (e.g, "Show only Action movies") and refine the score. |
| **vote_average** | **Quality Control:** Can display the vote average of the movie |
| **release_date** | **Temporal Filtering:** Enables recommendations based on time periods (e.g., "Movies from the 1990s"). |

### Run test
The module is built using an Object-Oriented (OO) approach to ensure easy integration.

```Bash
from clean_data import MovieCleaner

#Pipeline initialization and launch
cleaner = MovieCleaner()
cleaner.run_pipeline()
```
Result : 'movies_cleaned.csv'




## B) Vectorisation

### Models study : NLP (Natural Language Processing)
We studied different models before making a choice : 
- ELMo and Word2Vec : transform a word into a unique vector, but it might be confusing depending on the word (e.g : a bat -> the nocturnal animal or the baseball paddle)
- Text-to-Transform (T5) : can only traduce text input, it does not make sense for the purpose.
- Generative Pre-trained Transformer (GPT) : useful to understand and to categorize text input, but also to write text by watching the left word to predict the next one. 
- BERT : an encoder that analyzes text to perform recommendations or sentiment analysis.

This is why BERT is the best model for the project. Indeed, it captures the nuances of natural language, and uses bidirectional meaning. 

### Models choice : BERT
They are different BERT models : 
- To maximize precision : all-mpnet-base-v2: a little slow but efficient
multi-qa-mpnet-base-dot-v1:  optimized to answer user questions and to find documentation.
- For French dataset : 
paraphrase-mutlilingue-MiniLM-L12-v2 and distiluse-base-multilingual-cased-v1
- For speed (like API) : 
paraphrase-albert-small-v2 and all-MiniLM-L12-v2, where ALBERT is model has fewer parameters (6 instead of 12), which makes it less robust.

For the project we choose all-MiniLM-L12-v2 which is fast and useful for our English database. 

### Main functionalities:
Deep Learning model initialization thanks to SentenceTransformer
Synopsis vectorization :
Reading the CSV file and loading the synopses
Encoding the synopses
Saving the vectorized synopses to a new .npy file


### Run test
To test if the NLP model is correctly loaded and see how BERT "sees" a movie synopsis, you can run the following test script. It demonstrates the transformation of raw text into a 384-dimensional vector.

1. Create a file named test_encoding.py:
```Bash
from MovieEncoder import MovieEncoder
import numpy as np


# 1. Initialize the encoder (loads the BERT model)
encoder = MovieEncoder()


# 2. Define a test synopsis
synopsis = "A space pirate fighting aliens in a galaxy far away."


# 3. Generate the embedding using the model inside our class
print(f"Vectorizing: '{synopsis}'...")
vector = encoder.model.encode([synopsis])


# 4. Display the results
print(f"✅ Success! Vector shape: {vector.shape}")
print(f"First 5 numerical values of the vector: \n{vector[0][:5]}")

```

2. Expected Output in your terminal:
[INFO] Chargement du modèle all-MiniLM-L6-v2...
Vectorizing: 'A space pirate fighting aliens in a galaxy far away.'...
✅ Success! Vector shape: (1, 384)
First 5 numerical values of the vector: 
[-0.04049212  0.13097127 -0.04528323 -0.02100811 -0.06861791]



## C) Recommendation
This feature compares the distance between the vectors table and the user text. All the tables are numpy.

### Main functionalities : 
- Similarity computing: for each movie we calculate the Cosine Similarity between this movie vector and the user text vector. 
The result is among 1 and -1.
		    1 Perfect match |-1 Perfect mismatch | 0 No correlation


- Retrieving movie information:
Since the movie array and the vector array have the same length, we can retrieve each movie’s info (titles, type) by comparing their corresponding positions.

```Bash
titre = df.iloc[i]["title"]  #iloc => lines searching (pandas)
```
- Top 3 movie selection :The list of movies with their similarity scores is filtered in descending order, allowing us to obtain the “Top Movie Recommendations” table.

```Bash
best_similarite_films.sort(key=lambda x:x[2], reverse=True) #tri ordre décroissant "2" place du score
top = best_similarite_films[:3] #on affiche les deux meilleurs films
```

### Run test : 

```Bash
import numpy as np
import pandas as pd




#dataset & files
df = pd.read_csv('movies_cleaned.csv')
films_vecteurs = np.load("synopsis_embeddings.npy")


#SIMULATION
utilisateur_vecteur = np.random.rand(films_vecteurs.shape[1]).astype('float32')


best_similarite_films= [] #va contenir le titre et le score synopsis


similarites = []


for i in range(len(films_vecteurs)):
    score = np.dot(utilisateur_vecteur, films_vecteurs[i]) / (np.linalg.norm(utilisateur_vecteur) * np.linalg.norm(films_vecteurs[i]))
    similarites.append(score)
    titre = df.iloc[i]["title"]  #iloc => recherche de la ligne dans l'index (pandas)
    genre = df.iloc[i]["genres"]
    best_similarite_films.append((titre, genre, float(score)))


best_similarite_films.sort(key=lambda x:x[2], reverse=True) #tri ordre décroissant "2" place du score
top = best_similarite_films[:3] #on affiche les deux meilleurs films
print("Top 3 recommandations :")
for film in top:
    print(film[0], ":", film[1], film[2])

```



