import numpy as np
import pandas as pd


#chargement du dataset
df = pd.read_csv('movies_cleaned.csv')
#chargement du fichier contenant les vecteurs des synopsis
films_vecteurs = np.load("synopsis_embeddings.npy")



#SIMULATION entrée utilisateur
utilisateur_vecteur = np.random.rand(films_vecteurs.shape[1]).astype('float32')

best_similarite_films= [] #va contenir le titre et le score synopsis

#contient les scores de similarités de chaque film avec le vecteur utilisateur
similarites = []

#parcourir les films + vecteurs de la base de données
for i in range(len(films_vecteurs)):
    score = np.dot(utilisateur_vecteur, films_vecteurs[i]) / (np.linalg.norm(utilisateur_vecteur) * np.linalg.norm(films_vecteurs[i])) 
    similarites.append(score)


#AFFICHAGE 
    titre = df.iloc[i]["title"]  #iloc => recherche de la ligne dans l'index (pandas)
    genre = df.iloc[i]["genres"]
    best_similarite_films.append((titre, genre, float(score)))


#AFFICHAGE
best_similarite_films.sort(key=lambda x:x[2], reverse=True) #tri ordre décroissant "2" place du score
top = best_similarite_films[:3] #on affiche les deux meilleurs films
print("Top 3 recommandations :")
for film in top:
    print(film[0], ":", film[1], film[2])