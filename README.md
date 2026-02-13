# 🎬💙 Moveetic ! Movies Recommandation 
*Master 1, ISEN, January to February 2026, Teacher: Cyril Barrelet*

Analyzing movie synopses based on user input,
Using : NLP (BERT), Cosine Similarity and an API interface (FastAPI).


## 📖 1) Project Summary
Mooveetic is an intelligent movie recommendation engine that uses Natural Language Processing (NLP) and Deep Learning. Our code can understand the meaning of movie plots to suggest films with similar themes and narratives.

The diagram below shows how our movie recommendation system based on embeddings works.

<img width="503" height="302" alt="tradtab" src="https://github.com/user-attachments/assets/c2c4b529-f6c4-4fc6-8471-5d10c5193f36" />


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

## D) Interface API (FastAPI)

The API is Moveetic's communication interface. It acts as a conductor, connecting data cleaning, BERT encoding, and the search engine to provide real-time recommendations through HTTP requests.

### Main functionalities : 
- Automation conductor : First, the API checks the existence of the necessary files (cleaned .csv and vectorized .npy). If any files are missing, it automatically starts the cleaning and vectorization pipelines.
- Interactive documentation : Thanks to Swagger UI, the API creates a self-generated documentation. 
- Real-time processing : receives a title and a synopsis and transforms them in a vector. Then, it asks the dataset to return the Top 3 recommendation in JSON format.
- Architecture : uses Pydantic for input data validation and Uvicorn as a high-performance ASGI server.
The API uses MovieCleaner,  MovieEncoder and MovieSearcher; following the single responsibility principle in object-oriented programming (OOP).


### Run test
The API is designed to be easy to use. Here’s how to start the server and make a request.

1. Launching serveur
``` bash
#In the terminal
python main.py
```
2. Make a request
Option A : Interactive user interface (Swagger UI)
This method is perfect to visualize and to compute and to run the program.
Action : Launch the server http://127.0.0.1:8000/docs.
How to use : Click on "Try it out", fill in title and synopsis, then click on "Execute".
Benefits : No command-line usage is needed, and documentation is accessible in real time.

<img width="533" height="350" alt="1" src="https://github.com/user-attachments/assets/9e66523e-94db-45ea-9012-39cca2d91b42" />


### Example to Copy/past 
Title: Thor: The Dark World ; 
Synopsis : Thor fights to restore order across the cosmos… but an ancient race led by the vengeful Malekith returns to plunge the universe back into darkness. Faced with an enemy that even Odin and Asgard cannot withstand, Thor must embark on his most perilous and personal journey yet, one that will reunite him with Jane Foster and force him to sacrifice everything to save us all.

Result: 

<img width="750" height="207" alt="2" src="https://github.com/user-attachments/assets/a68216a1-abdb-47b5-b52b-70b7e93bcddd" />


Option B : URL calling (HTTP request)
Allows Moveetic’s integration into any application (website/ mobile app).
Request : 
```Bash
http://127.0.0.1:8000/analyze?title=Inception&synopsis=A+thief+who+enters+the+dreams+of+others.
```

Result : The server returns a JSON object in raw form, structured to be easily read by another program.

<img width="756" height="64" alt="3" src="https://github.com/user-attachments/assets/3057670f-c82a-4897-bfe8-a3b331af89ea" />



## E) Kmeans and T-SNE model :

To structure our data, we used synopses vectors (embeddings)

1. Optimal number of clusters
Several K-Means models with different numbers of clusters were used. The evaluation by the silhouette score designated K=2 as the optimal configuration. However, the score obtained is 0.025, which indicates a very low separation between the groups.

2. Score meaning
A score of 0.025 suggests that the clusters overlap. 
Some reasons may explain this:
Semantic Complexity: Movie synopses often deal with overlapping themes (love, death, adventure), making the boundaries blurry.
“The curse of dimensionality” : Working in a 384-dimensional space makes K-Means clustering less effective (Euclidean distances).
Data Density: BERT all movies are semantically close which prevents the formation of isolated spots.

3. Dimensionality reduction and visualization
To see the results, we used t-SNE. It can project our 384 dimension vectors into a 2D space. This visualization shows the two clusters and the overlap between them.
Visualization:

<img width="410" height="311" alt="4" src="https://github.com/user-attachments/assets/e09db133-0e85-4ccc-9a49-dc89f3fc2b90" />


This graphic shows how the synopses are correlated, and the two colors are mixed in the center. The low silhouette score of 0.025 confirms that the two algorithmically identified groups are poorly separated. Indeed, a movie has different topics :  Action-Drama, Horror-Comedy; the boundaries are blurred. Movies are a continuous set rather than completely isolated categories.







