import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class MovieEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialise le modèle de Deep Learning. 
        Le modèle n'est chargé en mémoire qu'une seule fois.
        """
        print(f"[INFO] Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name)

    def vectorize_csv(self, input_csv, output_npy=None):
        """
        Lit un CSV, vectorise les synopsis et sauvegarde (optionnel) en .npy
        """
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Le fichier {input_csv} est introuvable.")

        # Chargement des données
        df = pd.read_csv(input_csv)
        
        # Sécurité : on s'assure que la colonne 'overview' ne contient pas de vide
        synopses = df['overview'].fillna("").tolist()
        
        print(f"[INFO] Vectorisation de {len(synopses)} ligne(s)...")
        embeddings = self.model.encode(synopses, show_progress_bar=(len(synopses) > 1))

        # Sauvegarde si un chemin est fourni
        if output_npy:
            np.save(output_npy, embeddings)
            print(f"[SUCCÈS] Embeddings sauvegardés dans : {output_npy}")

        return embeddings