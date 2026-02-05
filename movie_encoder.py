# movie_encoder.py

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class MovieEncoder:
    """
    Classe responsable de la transformation des synopsis en vecteurs numériques.
    Elle utilise des modèles de Deep Learning (Sentence-BERT) pour capturer
    le sens sémantique des textes.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialise le modèle de langage.
        Le modèle n'est chargé en mémoire qu'une seule fois à l'instanciation.
        """
        print(f"[INFO] Chargement du modèle NLP : {model_name}...")
        self.model = SentenceTransformer(model_name)

    def vectorize_csv(self, input_csv: str, output_npy: str = None) -> np.ndarray:
        """
        Lit un fichier CSV, convertit les résumés en vecteurs (embeddings)
        et les sauvegarde éventuellement dans un fichier binaire .npy.
        """
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Le fichier {input_csv} est introuvable.")

        # Chargement des données avec pandas
        df = pd.read_csv(input_csv)
        
        # Sécurité : on remplace les valeurs manquantes par du texte vide
        # pour éviter que l'encodeur ne plante
        descriptions = df['overview'].fillna("").tolist()
        
        count = len(descriptions)
        print(f"[INFO] Début de la vectorisation de {count} ligne(s)...")
        
        # Génération des vecteurs
        # La barre de progression ne s'affiche que s'il y a plusieurs films
        embeddings = self.model.encode(
            descriptions, 
            show_progress_bar=(count > 1)
        )

        # Sauvegarde sur le disque si un chemin de sortie est spécifié
        if output_npy:
            np.save(output_npy, embeddings)
            print(f"[SUCCÈS] Vecteurs sauvegardés dans : {output_npy}")

        return embeddings