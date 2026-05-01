import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import time

# 1. Charger la clé API
load_dotenv()
API_KEY = os.getenv("PRIM_API_KEY")

if not API_KEY:
    print("Erreur : La clé API n'a pas été trouvée.")
    exit()

headers = {"apikey": API_KEY, "Accept": "application/json"}

print("Connexion à l'API d'Île-de-France Mobilités en cours...")

toutes_les_perturbations = []
page_actuelle = 0
items_par_page = 1000
continuer_aspiration = True

# --- ÉTAPE 1 : ASPIRATION TOTALE ---
while continuer_aspiration:
    URL = f"https://prim.iledefrance-mobilites.fr/marketplace/v2/navitia/disruptions?count={items_par_page}&start_page={page_actuelle}"

    print(f"Aspiration de la page {page_actuelle}...")
    response = requests.get(URL, headers=headers)

    if response.status_code == 200:
        data = response.json()

        if "disruptions" in data and len(data["disruptions"]) > 0:
            perturbations = data["disruptions"]
            toutes_les_perturbations.extend(perturbations)

            if len(perturbations) < items_par_page:
                continuer_aspiration = False
            else:
                page_actuelle += 1
                time.sleep(0.5)
        else:
            continuer_aspiration = False
    else:
        print(f"Erreur {response.status_code}")
        continuer_aspiration = False

print(f"\n🎯 Total brut récupéré : {len(toutes_les_perturbations)} perturbations.")

# --- ÉTAPE 2 : EXTRACTION (TOUS RÉSEAUX) ---
lignes_extraites = []

for pert in toutes_les_perturbations:
    pert_id = pert.get("id", "")
    statut = pert.get("status", "")
    cause = pert.get("cause", "")
    severite = pert.get("severity", {}).get("effect", "")
    periods = pert.get("application_periods", [])
    debut = periods[0].get("begin", "") if periods else ""
    fin = periods[0].get("end", "") if periods else ""
    messages = pert.get("messages", [])
    message_texte = messages[0].get("text", "").replace("\n", " ") if messages else ""

    impacts = pert.get("impacted_objects", [])

    for impact in impacts:
        pt_object = impact.get("pt_object", {})
        embedded_type = pt_object.get("embedded_type", "")

        nom_ligne = mode = nom_gare = id_gare = id_ligne = ""

        if embedded_type == "line":
            line = pt_object.get("line", {})
            mode = line.get("commercial_mode", {}).get("name", "Inconnu")
            id_ligne = line.get("id", "")
            nom_ligne = line.get("name", "")

            impacted_stops = impact.get("impacted_stops", [])
            if impacted_stops:
                for stop in impacted_stops:
                    stop_point = stop.get("stop_point", {})
                    lignes_extraites.append(
                        {
                            "id_perturbation": pert_id,
                            "statut": statut,
                            "cause": cause,
                            "severite": severite,
                            "mode": mode,
                            "nom_ligne": nom_ligne,
                            "id_ligne_IDFM": id_ligne,
                            "nom_gare": stop_point.get("name", ""),
                            "id_gare_IDFM": stop_point.get("id", ""),
                            "debut": debut,
                            "fin": fin,
                            "message": message_texte,
                        }
                    )
                continue
            else:
                nom_gare = "Toute la ligne"

        elif embedded_type in ["stop_area", "stop_point"]:
            stop_obj = pt_object.get(embedded_type, {})
            nom_gare = stop_obj.get("name", "")
            id_gare = stop_obj.get("id", "")
            mode = "Gare/Arrêt"

        # On ajoute tout sans filtre de mode
        lignes_extraites.append(
            {
                "id_perturbation": pert_id,
                "statut": statut,
                "cause": cause,
                "severite": severite,
                "mode": mode,
                "nom_ligne": nom_ligne,
                "id_ligne_IDFM": id_ligne,
                "nom_gare": nom_gare,
                "id_gare_IDFM": id_gare,
                "debut": debut,
                "fin": fin,
                "message": message_texte,
            }
        )

# --- ÉTAPE 3 : CRÉATION DU DATAFRAME ET NETTOYAGE DES PREFIXES ---
df_nouveau = pd.DataFrame(lignes_extraites)

if not df_nouveau.empty:
    # Nettoyage systématique des préfixes pour les jointures
    df_nouveau["id_ligne_IDFM"] = (
        df_nouveau["id_ligne_IDFM"].astype(str).str.replace("line:", "", regex=False)
    )
    df_nouveau["id_gare_IDFM"] = (
        df_nouveau["id_gare_IDFM"]
        .astype(str)
        .str.replace("stop_point:", "", regex=False)
    )
    df_nouveau["id_gare_IDFM"] = (
        df_nouveau["id_gare_IDFM"]
        .astype(str)
        .str.replace("stop_area:", "", regex=False)
    )
    # Remplacer les valeurs "None" ou vides par du vrai vide
    df_nouveau.replace("None", "", inplace=True)

# --- ÉTAPE 4 : UPSERT (ACCUMULATION) ---
nom_fichier = "data/perturbations_tous_reseaux.csv"

if os.path.exists(nom_fichier):
    df_existant = pd.read_csv(nom_fichier)
    df_final = pd.concat([df_existant, df_nouveau]).drop_duplicates(
        subset=["id_perturbation", "id_gare_IDFM", "id_ligne_IDFM"], keep="last"
    )
else:
    df_final = df_nouveau

# Sauvegarde finale
df_final.to_csv(nom_fichier, index=False, encoding="utf-8")

print(
    f"✅ Terminé ! Le fichier '{nom_fichier}' a été mis à jour (Tous réseaux inclus)."
)
print(f"Vérification : {len(df_final)} lignes enregistrées sans préfixe 'line:'.")
