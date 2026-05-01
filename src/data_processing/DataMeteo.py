import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. NETTOYAGE (Votre code)
# ==========================================
print("Chargement du fichier météo...")
# Note : assurez-vous que le dossier 'data/' existe
df_meteo = pd.read_csv("data/open-meteo.csv", skiprows=3)

colonnes_propres = {
    "time": "date_heure",
    "temperature_2m (°C)": "temperature",
    "relative_humidity_2m (%)": "humidite",
    "precipitation (mm)": "precipitation",
    "weather_code (wmo code)": "code_meteo",
    "wind_speed_10m (km/h)": "vitesse_vent",
}
df_meteo = df_meteo.rename(columns=colonnes_propres)
df_meteo = df_meteo.dropna()
df_meteo["date_heure"] = pd.to_datetime(df_meteo["date_heure"])

print(f"✅ Nettoyage terminé : {len(df_meteo)} lignes valides.")

# ==========================================
# 2. FEATURE ENGINEERING (Spécifique Projet Mobilité)
# ==========================================
# Extraction des périodes clés
df_meteo["heure"] = df_meteo["date_heure"].dt.hour
df_meteo["date"] = df_meteo["date_heure"].dt.date

# Identification des heures de pointe (Impact trafic automobile)
df_meteo["est_heure_pointe"] = df_meteo["heure"].isin([7, 8, 9, 17, 18, 19])

# Condition "Mobilité Douce OK" (Favorise le vélo/marche)
# Critères : Pas de pluie, Température entre 10 et 25°C, Vent < 20km/h
df_meteo["mobilite_douce_ok"] = (
    (df_meteo["precipitation"] == 0)
    & (df_meteo["temperature"].between(10, 25))
    & (df_meteo["vitesse_vent"] < 20)
)

# ==========================================
# 3. EXPLORATION GRAPHIQUE (EDA)
# ==========================================
sns.set_theme(style="whitegrid")

# --- Graphique A : Fenêtres d'opportunité Vélo ---
plt.figure(figsize=(10, 5))
prob_mobilite = df_meteo.groupby("heure")["mobilite_douce_ok"].mean() * 100
sns.lineplot(
    x=prob_mobilite.index, y=prob_mobilite.values, marker="o", color="forestgreen"
)
plt.fill_between(
    prob_mobilite.index, prob_mobilite.values, color="forestgreen", alpha=0.1
)
plt.title("Probabilité de conditions favorables à la mobilité douce (%)", fontsize=12)
plt.xlabel("Heure de la journée")
plt.ylabel("% de chance (Météo idéale)")
plt.xticks(range(24))
plt.savefig("data/analyse_velo.png")
plt.show()

# --- Graphique B : Risque de congestion (Pluie aux heures de pointe) ---
plt.figure(figsize=(10, 5))
pluie_rush = (
    df_meteo[df_meteo["est_heure_pointe"]].groupby("heure")["precipitation"].mean()
)
sns.barplot(x=pluie_rush.index, y=pluie_rush.values, palette="Blues_d")
plt.title("Intensité de pluie aux heures de pointe (Facteur de bouchons)", fontsize=12)
plt.ylabel("Précipitations moyennes (mm)")
plt.savefig("data/risque_trafic.png")
plt.show()

# --- Graphique C : Carte thermique des températures ---
pivot_temp = df_meteo.pivot_table(index="heure", columns="date", values="temperature")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_temp, cmap="YlOrRd")
plt.title("Confort thermique des usagers (Heures vs Jours)", fontsize=12)
plt.savefig("data/heatmap_confort.png")
plt.show()

# ==========================================
# 4. SAUVEGARDE
# ==========================================
df_meteo.to_csv("data/meteo_propre.csv", index=False, sep=";")
print("\n✅ Fichier enrichi sauvegardé sous 'data/meteo_propre.csv'")
print("📊 Graphiques d'exploration générés dans le dossier 'data/'")
