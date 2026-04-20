import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# Configuration de la page
st.set_page_config(page_title="Observatoire Mobilité Paris", layout="wide")


@st.cache_data
def load_data():
    fichier = 'analyse_perturbations_categories.csv'
    if not os.path.exists(fichier):
        st.error(f"Le fichier '{fichier}' est introuvable.")
        return pd.DataFrame()

    df = pd.read_csv(fichier, sep=';')
    # Conversion des dates
    df['debut_dt'] = pd.to_datetime(df['debut'], format='%Y%m%dT%H%M%S', errors='coerce')
    df['fin_dt'] = pd.to_datetime(df['fin'], format='%Y%m%dT%H%M%S', errors='coerce')

    # Sécurité : on s'assure que les noms de lignes sont bien du texte (ex: la ligne 9 ne doit pas être un chiffre)
    if 'nom_ligne' in df.columns:
        df['nom_ligne'] = df['nom_ligne'].astype(str)

    return df


df = load_data()

# --- BARRE LATÉRALE (FILTRES GLOBAUX) ---
st.sidebar.header("🔍 Filtres globaux")

# 1. Filtre par Type de Ligne (Mode)
if 'mode_x' in df.columns:
    modes_disponibles = sorted(df['mode_x'].dropna().unique().tolist())
    modes_selectionnes = st.sidebar.multiselect(
        "Type de transport",
        options=modes_disponibles,
        default=['Métro', 'RER']
    )
else:
    modes_selectionnes = []

# --- NOUVEAU : Filtre par Ligne (Cascade) ---
# On filtre temporairement pour ne proposer que les lignes du mode sélectionné
df_temp = df[df['mode_x'].isin(modes_selectionnes)] if modes_selectionnes else df
lignes_disponibles = sorted(df_temp['nom_ligne'].dropna().unique().tolist())

lignes_selectionnees = st.sidebar.multiselect(
    "Filtrer par Ligne(s) spécifique(s)",
    options=lignes_disponibles,
    default=[],
    help="Laissez vide pour voir toutes les lignes du mode sélectionné."
)

# 3. Filtre par Sévérité
severites_dispo = df['severite'].unique()
severites_selectionnees = st.sidebar.multiselect(
    "Niveau de sévérité",
    options=severites_dispo,
    default=severites_dispo
)

# 4. Filtre par Catégorie (NLP)
categories_dispo = sorted(df['categorie_cause'].dropna().unique().tolist())
categories_selectionnees = st.sidebar.multiselect(
    "Catégorie de l'incident",
    options=categories_dispo,
    default=categories_dispo
)

# --- APPLICATION DU FILTRE GLOBAL ---
mask = (df['mode_x'].isin(modes_selectionnes)) & \
       (df['severite'].isin(severites_selectionnees)) & \
       (df['categorie_cause'].isin(categories_selectionnees))

# Si l'utilisateur a choisi une ou plusieurs lignes, on ajoute ça au filtre
if lignes_selectionnees:
    mask = mask & (df['nom_ligne'].isin(lignes_selectionnees))

df_filtered = df[mask].copy()

# --- CORPS DE L'INTERFACE ---
st.title("🗼 État du réseau de transport - Paris")

# Indicateurs clés (se mettent à jour dynamiquement)
col1, col2, col3 = st.columns(3)

# On compte les stations physiques uniques et non plus le nombre de lignes du tableau
with col1:
    stations_uniques = df_filtered['stop_name'].nunique() if not df_filtered.empty else 0
    st.metric("📍 Stations physiques impactées", stations_uniques)

with col2:
    st.metric("🚇 Lignes touchées", df_filtered['nom_ligne'].nunique())
with col3:
    st.metric("⚠️ Incidents uniques", df_filtered['id_perturbation'].nunique())

# --- CARTE INTERACTIVE ---
st.subheader("📍 Localisation des points rouges (Incidents)")
if not df_filtered.empty:
    # On centre sur Paris
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="cartodbpositron")
    for _, row in df_filtered.head(1000).iterrows():
        folium.CircleMarker(
            location=[row['stop_lat'], row['stop_lon']],
            radius=6,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=f"<b>Ligne {row['nom_ligne']}</b><br>{row['stop_name']}<br><i>{row['categorie_cause']}</i>",
        ).add_to(m)
    st_folium(m, width=1400, height=450)
else:
    st.warning("Aucune donnée disponible avec les filtres actuels.")

st.divider()

# --- SECTION : DÉTAILS ---
st.header("🔍 Détail des arrêts impactés")

if not df_filtered.empty:
    col_table, col_chart = st.columns([2, 1])

    with col_table:
        st.write("**Liste des arrêts subissant un incident :**")
        # On affiche le tableau en fonction des filtres globaux
        arrets_impactes = df_filtered[
            ['nom_ligne', 'stop_name', 'categorie_cause', 'severite', 'message_clean']].drop_duplicates().sort_values(
            ['nom_ligne', 'stop_name'])
        st.dataframe(arrets_impactes, use_container_width=True, height=300)

    with col_chart:
        st.write("**Répartition par catégorie :**")
        cause_counts = df_filtered.drop_duplicates(subset=['nom_ligne', 'stop_name'])['categorie_cause'].value_counts()
        st.bar_chart(cause_counts)
else:
    st.info("Aucune donnée à afficher dans le tableau.")

# --- ANALYSES GLOBALES ---
st.divider()
st.subheader("📊 Statistiques des données filtrées")

# NOUVEAU : Un seul graphique centré et lisible sur toute la largeur
st.write("**Top 10 des lignes les plus impactées (Incidents uniques)**")
incidents_par_ligne = df_filtered.groupby('nom_ligne')['id_perturbation'].nunique().sort_values(
    ascending=False).head(10)
st.bar_chart(incidents_par_ligne)

# Bouton de téléchargement
st.divider()
csv = df_filtered.to_csv(index=False, sep=';').encode('utf-8')
st.sidebar.download_button(
    "📥 Télécharger les données affichées",
    data=csv,
    file_name="export_paris_filtre.csv",
    mime="text/csv"
)