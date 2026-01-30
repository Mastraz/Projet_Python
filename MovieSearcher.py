import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieSearcher:
    def __init__(self, csv_path='movies_cleaned.csv', npy_path='synopsis_embeddings.npy'):
        """Charge les données une seule fois au démarrage."""
        print("[INFO] Chargement de la base de données et des vecteurs...")
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(npy_path)

    def find_similar_movies(self, query_vector, top_n=3):
        """Calcule la similarité cosinus et retourne les meilleurs matchs."""
        
        # Calcul de la similarité entre le vecteur utilisateur et TOUTE la matrice
        # scores sera un tableau contenant 4800+ scores de 0 à 1
        scores = cosine_similarity(query_vector, self.embeddings).flatten()

        # Récupération des indices des meilleurs scores
        # argsort trie par ordre croissant, on prend les derniers et on inverse
        top_indices = scores.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "title": self.df.iloc[idx]["title"],
                "genres": self.df.iloc[idx]["genres"],
                "similarity_score": round(float(scores[idx]), 3)
            })
            
        return results