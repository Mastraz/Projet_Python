import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

#Chargement du dataset
df = pd.read_csv('movies_cleaned.csv')
print(df.shape)
#Chargement du modèle de deep learning miniature (basé sur BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

#Etape de vectorisation des synopsis
#chaque synopsis est converti en un vecteur de 384 dimensions
embeddings = model.encode(df['overview'].tolist(), show_progress_bar=True)
print(embeddings.shape)

# Sauvegarde des embeddings dans un fichier .npy (fichier binaire NumPy) qui pourra être réutilisé ultérieurement
np.save('synopsis_embeddings.npy', embeddings)

# Affichage d'un aperçu des embeddings générés
print(embeddings[:3, :5])