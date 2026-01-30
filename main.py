from fastapi import FastAPI
import pandas as pd
from MovieCleaner import MovieCleaner
from MovieEncoder import MovieEncoder
from MovieSearcher import MovieSearcher
import os

# --- INITIALISATION ---
app = FastAPI(title="API Cinéma - Version Simple")
# Attention : cleaner() charge et nettoie tout le dataset à chaque lancement du serveur
cleaner = MovieCleaner()
encoder = MovieEncoder()
searcher = MovieSearcher()
# --- ETAPE 1 : Vectorisation de la base globale (au démarrage) ---
# On le fait une fois pour avoir le fichier .npy global
if not os.path.exists('synopsis_embeddings.npy'):
    print("[INIT] Première vectorisation de la base de données...")
    encoder.vectorize_csv('movies_cleaned.csv', output_npy='synopsis_embeddings.npy')

# --- LA ROUTE EN GET ---

@app.get("/analyze")
def analyze_movie(title: str, synopsis: str):
    # 1. On crée le fichier temporaire (indispensable pour votre structure actuelle)
    temp_filename = "temp_input.csv"
    pd.DataFrame({"title": [title], "overview": [synopsis]}).to_csv(temp_filename, index=False)
    
    # 2. On transforme ce texte en vecteur (le pôle NLP entre en action)
    # C'est ici que l'on appelle votre classe MovieEncoder
    query_embedding = encoder.vectorize_csv(temp_filename)
    
    # 3. Recherche des films similaires dans la base
    recommendations = searcher.find_similar_movies(query_embedding, top_n=3)
    
    # 4. Réponse finale
    return {
        "status": "Success",
        "movie_analyzed": title,
        "recommendations": recommendations
    }

# --- LANCEMENT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)