import pandas as pd
import ast
import os

# ==========================================
# VERSION FINALE : CLASS (POO)
# ==========================================

class MovieCleaner:
    """
    Classe responsable du chargement, nettoyage et sauvegarde du dataset de films.
    Elle gère les valeurs manquantes et le parsing des JSON.
    """

    def __init__(self, input_file="tmdb_5000_movies.csv", output_file="movies_cleaned.csv"):
        """Initialisation des chemins de fichiers."""
        self.input_file = input_file
        self.output_file = output_file
        self.df = None  # Le dataframe sera chargé plus tard

    def load_data(self):
        """Charge le fichier CSV en mémoire."""
        if not os.path.exists(self.input_file):
            print(f"[ERREUR] Le fichier '{self.input_file}' est introuvable.")
            return False
        
        self.df = pd.read_csv(self.input_file)
        print(f"[INFO] Données chargées : {self.df.shape[0]} films, {self.df.shape[1]} colonnes.")
        return True

    def _parse_json_genres(self, text):
        """
        Méthode interne pour transformer le JSON en liste.
        Ex: '[{"name": "Action"}]' -> ['Action']
        """
        try:
            data = ast.literal_eval(text)
            if isinstance(data, list):
                return [item['name'] for item in data]
            return []
        except (ValueError, SyntaxError):
            return []

    def clean_data(self):
        """Exécute toute la logique de nettoyage."""
        if self.df is None:
            print("[ATTENTION] Aucune donnée chargée.")
            return

        print("\n--- Début du Nettoyage ---")
        initial_count = len(self.df)

        # 1. Sélection des colonnes
        cols_to_keep = ['id', 'title', 'overview', 'genres', 'vote_average', 'release_date']
        self.df = self.df[cols_to_keep].copy()

        # 2. Suppression des NaN (Vide)
        self.df = self.df.dropna(subset=['overview', 'release_date'])
        print(f"[INFO] Lignes supprimées (données manquantes) : {initial_count - len(self.df)}")

        # 3. Parsing des genres
        print("[INFO] Transformation des genres (JSON -> Texte)...")
        # On applique la méthode _parse_json_genres sur chaque ligne
        self.df['genres_list'] = self.df['genres'].apply(self._parse_json_genres)
        # On convertit en string pour le CSV final
        self.df['genres'] = self.df['genres_list'].apply(lambda x: ", ".join(x))
        
        # Nettoyage des colonnes temporaires
        self.df = self.df.drop(columns=['genres_list'])

        print("[SUCCES] Nettoyage terminé.")

    def save_data(self):
        """Sauvegarde le résultat final."""
        if self.df is not None:
            self.df.to_csv(self.output_file, index=False)
            print(f"\n[SUCCES] Fichier sauvegardé : {self.output_file}")
            print(f"[INFO] Colonnes finales : {self.df.columns.tolist()}")

    def run_pipeline(self):
        """
        Méthode 'Façade' : Lance tout le processus d'un coup.
        C'est cette méthode que les autres scripts appelleront.
        """
        print(f"[INFO] Démarrage du pipeline de nettoyage...")
        if self.load_data():
            self.clean_data()
            self.save_data()

# --- Zone d'exécution (Ce qui se passe quand on lance le fichier directement) ---
if __name__ == "__main__":
    # On instancie la classe
    cleaner = MovieCleaner()
    # On lance le traitement
    cleaner.run_pipeline()


# ==========================================
# ARCHIVES / HISTORIQUE
# (Ce code ne s'exécute pas : c'est le code original procédural conservé pour référence)
# ==========================================
"""
import pandas as pd
import os
import ast

# Chargement du fichier CSV
csv_path = "tmdb_5000_movies.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Fichier chargé !")
    print(f"Dimensions : {df.shape[0]} films et {df.shape[1]} colonnes.")
    
    # 2. Affichage des 3 premières lignes pour voir la tête des données
    print("\n--- Aperçu des données ---")
    print(df.head(3))
else:
    print("Fichier introuvable. Vérifie qu'il est bien dans le dossier.")

# Analyse des valeurs manquantes    

print("\n--- Analyse des valeurs manquantes ---")
missing_values = df.isnull().sum() # compte les cases vides (null) par colonne
print(missing_values)

# Petit calcul pour voir le % de perte si on nettoie
total_films = len(df)
missing_overview = missing_values['overview']
print(f"\nFilms sans résumé : {missing_overview}")

print("\n--- Nettoyage ---")

# sélectionne uniquement les colonnes utiles pour le projet
cols = ['id', 'title', 'overview', 'genres', 'vote_average', 'release_date']
# crée une copie propre du tableau avec juste ces colonnes
df = df[cols].copy()

# supprime les lignes vides (Les 3 sans résumé + le 1 sans date)
initial_count = len(df)
df = df.dropna(subset=['overview', 'release_date'])
final_count = len(df)

print(f"Films avant nettoyage : {initial_count}")
print(f"Films après nettoyage : {final_count}")
print(f"Films supprimés : {initial_count - final_count}")

print("\n--- Vérification du dataset propre ---")
print(df.head())
print(f"\nColonnes restantes : {df.columns.tolist()}")


# Nettoyage de la colonne 'genres'
print("\n--- Transformation des Genres ---")

# Fonction qui lit le texte comme du code
def nettoyer_json(texte):
    try:
        # Convertit le texte "[{'name': 'Action'}]" en liste Python
        liste = ast.literal_eval(texte)
        # On garde uniquement les noms des genres
        return [item['name'] for item in liste]
    except:
        return [] # En cas d'erreur, on renvoie une liste vide

# applique la transformation sur toute la colonne
df['genres_clean'] = df['genres'].apply(nettoyer_json)

# VÉRIFICATION
print(f"AVANT (dirty) : {df.iloc[0]['genres']}")
print(f"APRÈS (clean) : {df.iloc[0]['genres_clean']}")

# crée une version "texte simple" pour l'affichage (ex: "Action, Adventure")
# Cela servira aussi si quelqu'un ouvre le CSV dans Excel
df['genres_str'] = df['genres_clean'].apply(lambda x: ", ".join(x))

# supprime la vieille colonne 'genres' (dirty) pour ne pas alourdir le fichier
df = df.drop(columns=['genres'])

# supprime la colonne 'genres_clean' (la liste) car on a déjà la version texte
df = df.drop(columns=['genres_clean'])

# renomme 'genres_str' en 'genres' pour que ce soit propre
df = df.rename(columns={'genres_str': 'genres'})

# Vérification finale avant export
print("\n--- Vérification du dataset propre ---")
print("Colonnes finales :", df.columns.tolist())
print(df.head())

print("\n--- Sauvegarde du fichier ---")
output_file = "movies_cleaned.csv"
df.to_csv(output_file, index=False)

print(f"\nFichier sauvegardé : {output_file}")
"""