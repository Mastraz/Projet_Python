# 🎬💙 Moveetic ! Movies Recommandation 
*Master 1, ISEN, January to February 2026, Teacher: Cyril Barrelet*

Analyzing movie synopses based on user input,
Using : NLP (BERT), Cosine Similarity and an API interface (FastAPI).


## 📖 1) Project Summary
Mooveetic is an intelligent movie recommendation engine that uses Natural Language Processing (NLP) and Deep Learning. Our code can understand the meaning of movie plots to suggest films with similar themes and narratives.

### Key Features:
- Automated Data Cleaning: preprocessing of the TMDB dataset, including JSON parsing and handling missing values.
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
