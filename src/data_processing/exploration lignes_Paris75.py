import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour un affichage propre
plt.style.use("ggplot")

# 1. Chargement du dataset nettoyé
file_path = "processed/arrets_lignes_paris_75.csv"
df = pd.read_csv(file_path, sep=";")

print("--- APERÇU DES DONNÉES ---")
print(df.info())
print("\n--- STATISTIQUES DESCRIPTIVES ---")
print(df.describe())

# --- 2. ANALYSE DE LA RÉPARTITION PAR LIGNE ---
print("\n--- TOP 10 DES LIGNES AVEC LE PLUS D'ARRÊTS ---")
top_lignes = df["route_long_name"].value_counts().head(10)
print(top_lignes)

plt.figure(figsize=(12, 6))
top_lignes.plot(kind="barh", color="teal")
plt.title("Top 10 des lignes les plus denses à Paris")
plt.xlabel("Nombre d'arrêts")
plt.gca().invert_yaxis()
plt.show()

# --- 3. ANALYSE GÉOGRAPHIQUE PAR ARRONDISSEMENT ---
# On extrait l'arrondissement du code_insee (ex: 75101 -> 1er)
df["arrondissement"] = df["code_insee"].astype(str).str[-2:]

print("\n--- RÉPARTITION PAR ARRONDISSEMENT (CODE INSEE) ---")
districts = df["arrondissement"].value_counts().sort_index()
print(districts)

plt.figure(figsize=(10, 5))
sns.barplot(x=districts.index, y=districts.values, palette="viridis")
plt.title("Densité des points de transport par arrondissement")
plt.xlabel("Derniers chiffres du Code INSEE (Arrondissements)")
plt.ylabel("Nombre d'arrêts")
plt.show()

# --- 4. VÉRIFICATION DES COORDONNÉES (BOÎTE À MOUSTACHES) ---
# Utile pour détecter des erreurs de saisie GPS (Outliers)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[["stop_lon", "stop_lat"]])
plt.title("Vérification de la cohérence des coordonnées GPS")
plt.show()
