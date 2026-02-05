# movie_searcher.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieSearcher:
    """
    Classe chargée de comparer un vecteur de requête avec la base de données.
    Elle utilise la similarité cosinus pour identifier les films dont
    le sens sémantique est le plus proche du synopsis fourni.
    """

    def __init__(self, csv_path: str = 'movies_cleaned.csv', 
                 npy_path: str = 'synopsis_embeddings.npy'):
        """
        Initialisation du moteur de recherche.
        Charge le dataset nettoyé et la matrice de vecteurs en mémoire.
        """
        print("[INFO] Chargement de la base de données et des vecteurs...")
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(npy_path)

    def find_similar_movies(self, query_vector: np.ndarray, 
                            top_n: int = 3) -> list:
        """
        Calcule la similarité cosinus entre la requête et tous les films,
        puis retourne une liste des N meilleurs résultats.
        """
        # Calcul de la similarité entre la requête et toute la base de données.
        # 'scores' est un tableau contenant la proximité (0 à 1) pour chaque film.
        scores = cosine_similarity(query_vector, self.embeddings).flatten()

        # Récupération des indices des scores les plus élevés.
        # argsort trie par ordre croissant, on prend les derniers et on inverse.
        top_indices = scores.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            # On construit un dictionnaire pour chaque recommandation
            results.append({
                "title": self.df.iloc[idx]["title"],
                "genres": self.df.iloc[idx]["genres"],
                "similarity_score": round(float(scores[idx]), 3)
            })

        return results