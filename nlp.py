import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

#Chargement du dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Sélection des colonnes pertinentes
df = df[['id', 'title', 'overview']]

# Suppression des lignes avec des valeurs manquantes dans la colonne 'overview'
df = df.dropna(subset=['overview'])

# Affichage des premières lignes du DataFrame
print(df.head())

# Initialisation du modèle de transformation de phrases
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encodage des synopsis des films
embeddings = model.encode(df['overview'].tolist(), show_progress_bar=True)

# Sauvegarde des embeddings dans un fichier .npy
np.save('synopsis_embeddings.npy', embeddings)

# Affichage d'un aperçu des embeddings générés
print(embeddings[:3, :5])