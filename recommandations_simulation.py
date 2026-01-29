import numpy as np
import faiss

# SIMULATION
titres = ["Interstellar", "Star Wars", "StarTrek"]

#SIMULATION VECTEURS DE FILMS
films_vecteurs = [
    [0.2, 0.1, 0.4, 0.4],   # Interstellar
    [0.1, 0.3, 0.5, 0.2],   # Star wars
    [0.4, 0.2, 0.1, 0.3]    # Star trek
]

#SIMULATION entrée utilisateur
titre_utilisateur = ["Mandalorian"]
utilisateur_vecteur = [0.15, 0.25, 0.45, 0.3]

similarite = 0;
best_similarite = -1;
best_film = ""
best_similarite_films= []

#parcourir les films + vecteurs de la base de données
for i in range(len(films_vecteurs)):
    similarite = np.dot(utilisateur_vecteur, films_vecteurs[i]) / (np.linalg.norm(utilisateur_vecteur) * np.linalg.norm(films_vecteurs[i])) 
    best_similarite_films.append((titres[i], similarite))  #remplissage de la liste de films similaires

    
best_similarite_films.sort(key=lambda x:x[1], reverse=True) #tri ordre décroissant
top = best_similarite_films[:2] #on affiche les deux meilleurs films
print("Top 2 recommandations :")

for film in top:
    print(film[0], ":", film[1])
    
    """#Recommandation de 1 film
    # recherche du film à la plus grande similarité (ici Star Wars)
    if similarite > best_similarite :
        best_similarite = similarite
        best_film = titres[i]
    
print(best_similarite)
print(best_film)"""

