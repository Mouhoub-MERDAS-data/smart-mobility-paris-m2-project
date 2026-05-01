import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
df_pert = pd.read_csv("data/perturbations_tous_reseaux.csv")
df_ref = pd.read_csv("processed/arrets_lignes_paris_75.csv", sep=";")

print("📊 Dimensions")
print("Perturbations :", df_pert.shape)
print("Référentiel :", df_ref.shape)

# ==========================================
# 2. APERÇU
# ==========================================
print("\n🔍 Aperçu df_pert")
print(df_pert.head())

print("\n🔍 Aperçu df_ref")
print(df_ref.head())

# ==========================================
# 3. INFOS
# ==========================================
print("\nℹ️ Infos df_pert")
print(df_pert.info())

print("\nℹ️ Infos df_ref")
print(df_ref.info())

# ==========================================
# 4. VALEURS MANQUANTES
# ==========================================
print("\n❗ Valeurs manquantes df_pert")
print(df_pert.isna().sum().sort_values(ascending=False))

print("\n❗ Valeurs manquantes df_ref")
print(df_ref.isna().sum().sort_values(ascending=False))

# ==========================================
# 5. POURCENTAGE MANQUANTS
# ==========================================
missing_pct = (df_pert.isna().sum() / len(df_pert)) * 100
print("\n📊 % valeurs manquantes")
print(missing_pct.sort_values(ascending=False))

# ==========================================
# 6. ANALYSE CATÉGORIELLE
# ==========================================
print("\n📊 Modes")
print(df_pert["mode"].value_counts())

print("\n📊 Sévérité")
print(df_pert["severite"].value_counts())

# ==========================================
# 7. VISUALISATIONS
# ==========================================
plt.figure(figsize=(8, 5))
df_pert["mode"].value_counts().plot(kind="bar")
plt.title("Répartition des modes")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df_pert, x="severite")
plt.title("Répartition de la sévérité")
plt.show()

# ==========================================
# 8. TEMPS
# ==========================================
df_pert["debut"] = pd.to_datetime(df_pert["debut"], errors="coerce")

plt.figure(figsize=(10, 5))
df_pert["debut"].dt.hour.value_counts().sort_index().plot()
plt.title("Distribution par heure")
plt.xlabel("Heure")
plt.show()

# ==========================================
# 9. RENOMMAGE
# ==========================================
df_pert = df_pert.rename(
    columns={"id_ligne_IDFM": "id_ligne", "id_gare_IDFM": "id_gare"}
)

# ==========================================
# 10. ENRICHISSEMENT
# ==========================================
mapping_gares = df_ref.drop_duplicates(subset=["id_gare"])[["id_gare", "id_ligne"]]
dict_mapping = dict(zip(mapping_gares["id_gare"], mapping_gares["id_ligne"]))

mask = df_pert["id_ligne"].isnull() & df_pert["id_gare"].notnull()
df_pert.loc[mask, "id_ligne"] = df_pert.loc[mask, "id_gare"].map(dict_mapping)

print(
    "\n✅ id_ligne manquants après enrichissement :", df_pert["id_ligne"].isna().sum()
)


# ==========================================
# 11. NETTOYAGE HTML
# ==========================================
def clean_html(text):
    if pd.isna(text):
        return ""
    return re.sub("<.*?>", " ", str(text)).strip()


df_pert["message_clean"] = df_pert["message"].apply(clean_html)


# ==========================================
# 12. CATÉGORISATION
# ==========================================
def get_category(text):
    if not text:
        return "Autre / Divers"

    text = text.lower()

    mapping = {
        "Sécurité / Sûreté": [
            "colis suspect",
            "bagage",
            "alarme",
            "police",
            "intrusion",
            "incendie",
        ],
        "Problème Technique": ["panne", "signalisation", "aiguillage", "caténaire"],
        "Travaux / Maintenance": ["travaux", "chantier", "maintenance"],
        "Voyageur / Affluence": ["malaise", "affluence"],
        "Trafic / Exploitation": ["régulation", "train supprimé"],
        "Météo / Environnement": ["vent", "pluie", "neige"],
    }

    for cat, mots in mapping.items():
        if any(m in text for m in mots):
            return cat

    return "Autre / Divers"


df_pert["categorie_cause"] = df_pert["message_clean"].apply(get_category)

print("\n📊 Catégories :")
print(df_pert["categorie_cause"].value_counts())

# ==========================================
# 13. ANALYSE CROISÉE
# ==========================================
cross_pct = (
    pd.crosstab(df_pert["mode"], df_pert["categorie_cause"], normalize="index") * 100
)

print("\n📊 Répartition (%)")
print(cross_pct.round(2))

cross_pct.plot(kind="bar", stacked=True, figsize=(12, 7), colormap="viridis")
plt.title("Répartition des causes (%)")
plt.ylabel("Pourcentage")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# ==========================================
# 14. NETTOYAGE FINAL
# ==========================================
df_pert = df_pert.dropna(subset=["message"])
df_pert = df_pert.drop_duplicates()

print("\n📊 Lignes finales :", df_pert.shape[0])

# ==========================================
# 15. SAUVEGARDE
# ==========================================
os.makedirs("processed", exist_ok=True)

output = "processed/perturbations_final.csv"

df_pert.to_csv(output, index=False, sep=";", encoding="utf-8-sig")

print("\n💾 Fichier sauvegardé :", output)
