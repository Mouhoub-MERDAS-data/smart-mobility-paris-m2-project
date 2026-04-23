import pandas as pd

# 1. Configuration des chemins
file_path = "data/arrets-lignes.csv"
output_file = "data/arrets_lignes_paris_75.csv"

# Définition du mapping pour uniformiser les modes
mapping_modes = {
    "RapidTransit": "rer",
    "LocalTrain": "transilien",
    "regionalRail": "transilien",
    "Subway": "metro",
    "Tramway": "tram",
}

try:
    # Chargement des données
    df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    print(f"✅ Fichier chargé : {len(df)} lignes.")

    # 2. Filtrage immédiat pour Paris (75)
    df["code_insee"] = df["code_insee"].astype(str)
    df_paris = df[df["code_insee"].str.startswith("75")].copy()

    # 3. Sélection, RENOMMAGE et MAPPING
    nouveaux_noms = {"id": "id_ligne", "stop_id": "id_gare"}

    # Liste des colonnes à conserver (on s'assure que 'mode' est présent pour le mapping)
    colonnes_a_garder = [
        "id",
        "route_long_name",
        "stop_id",
        "stop_name",
        "stop_lon",
        "stop_lat",
        "pointgeo",
        "nom_commune",
        "code_insee",
        "mode",
    ]

    # Application du filtre de colonnes et renommage
    df_final = df_paris[colonnes_a_garder].rename(columns=nouveaux_noms)

    # --- AJOUT DU MAPPING ICI ---
    # On crée une nouvelle colonne 'mode_standard' basée sur ton dictionnaire
    # .fillna(df_final['mode']) permet de garder la valeur d'origine si elle n'est pas dans le mapping (ex: Bus)
    df_final["mode_standard"] = (
        df_final["mode"].map(mapping_modes).fillna(df_final["mode"])
    )
    # ----------------------------

    # 4. Nettoyage des valeurs nulles
    df_final = df_final.dropna(subset=["id_ligne", "stop_name"])

    # 5. Sauvegarde
    df_final.to_csv(output_file, index=False, sep=";", encoding="utf-8")

    print(f"🚀 Nettoyage, renommage et mapping terminés !")
    print(f"📊 Nombre d'arrêts à Paris : {len(df_final)}")
    print(f"📍 Aperçu des modes : {df_final['mode_standard'].unique()}")

except Exception as e:
    print(f"❌ Une erreur est survenue : {e}")
