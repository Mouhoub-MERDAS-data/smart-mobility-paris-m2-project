"""
Dashboard Streamlit — Observatoire de la pollution du périphérique parisien.

Lance avec :
    streamlit run interface/streamlit_app.py

Pages :
    1. Accueil   — KPIs globaux, équipe
    2. Prédictions — démo interactive par segment
    3. Analyses  — historique, distribution erreurs, profils
    4. Modèle    — feature importance, benchmark, matrice confusion
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ajouter la racine du projet au sys.path pour que 'src' soit importable
# peu importe depuis quel répertoire Streamlit est lancé
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Observatoire Mobilité Paris",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"

SEGMENTS = [
    "Chap-Bagn", "Bagn-Berc", "Berc-Ital", "Ital-A6a",
    "A6a-Sevr",  "Sevr-Aute", "Aute-Mail", "Mail-Chap",
]

SEUILS     = {"NO2": 40.0, "PM10": 40.0, "PM25": 25.0}
COLORS     = {"NO2": "#E63946", "PM10": "#F4A261", "PM25": "#2A9D8F"}

# ---------------------------------------------------------------------------
# Chargement des données et du modèle (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_pollution_history() -> pd.DataFrame:
    """Charge toutes les données de pollution disponibles."""
    frames = []
    for pollutant in ["NO2", "PM10", "PM25"]:
        for year in ["2024", "2025", "2026"]:
            candidates = list(DATA_DIR.glob(f"{year}*{pollutant}*"))
            if candidates:
                df = pd.read_csv(candidates[0], parse_dates=["time"])
                df["pollutant"] = pollutant
                frames.append(df)
    if not frames:
        st.error("⚠️ Aucun fichier de pollution trouvé dans data/")
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    df_long = raw.melt(
        id_vars=["time", "pollutant"],
        value_vars=[c for c in raw.columns if c in SEGMENTS],
        var_name="segment",
        value_name="value",
    )
    df_long = df_long.sort_values(["pollutant", "segment", "time"]).reset_index(drop=True)
    
    # Ajouter les colonnes temporelles nécessaires pour le Streamlit
    df_long["hour"]        = df_long["time"].dt.hour
    df_long["day_of_week"] = df_long["time"].dt.dayofweek
    df_long["day_name"]    = df_long["time"].dt.day_name()
    df_long["month"]       = df_long["time"].dt.month
    df_long["year"]        = df_long["time"].dt.year
    df_long["is_weekend"]  = (df_long["day_of_week"] >= 5).astype(int)
    
    return df_long


@st.cache_resource
def load_model(pollutant: str = "NO2", version: str = "v2"):
    """Charge le modèle XGBoost entraîné."""
    import pickle
    for v in [version, "v1"]:  # fallback sur v1 si v2 absent
        path = MODELS_DIR / f"xgboost_{pollutant.lower()}_{v}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model, v
    return None, None


@st.cache_data
def load_metrics(pollutant: str = "NO2", version: str = "v2") -> list | None:
    """Charge les métriques sauvegardées."""
    for v in [version, "v1"]:
        path = MODELS_DIR / f"metrics_{pollutant.lower()}_{v}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def sidebar() -> str:
    with st.sidebar:
        st.title("🚦 Observatoire\nMobilité Paris")
        st.divider()
        page = st.radio(
            "Navigation",
            ["🏠 Accueil", "📈 Prédictions", "🔬 Analyses", "⚙️ Modèle"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Mastère Big Data & IA — SUP DE VINCI 2025")
    return page


# ---------------------------------------------------------------------------
# PAGE 1 : Accueil
# ---------------------------------------------------------------------------

def page_accueil(df: pd.DataFrame):
    st.title("🚦 Observatoire de la Mobilité Urbaine — Périphérique Parisien")
    st.markdown(
        "Prédiction horaire des concentrations de polluants (NO₂, PM10, PM2.5) "
        "sur les 8 segments du boulevard périphérique, à partir de données Airparif + météo."
    )

    col1, col2, col3, col4 = st.columns(4)

    if not df.empty:
        total_lignes = len(df)
        n_segments   = df["segment"].nunique()
        periode_debut = df["time"].min().strftime("%b %Y")
        periode_fin   = df["time"].max().strftime("%b %Y")
        no2_mean      = df[df["pollutant"] == "NO2"]["value"].mean()

        col1.metric("Lignes de données", f"{total_lignes:,}")
        col2.metric("Segments couverts", n_segments)
        col3.metric("Période", f"{periode_debut} → {periode_fin}")
        col4.metric("NO₂ moyen", f"{no2_mean:.1f} µg/m³")

    st.divider()

    # Aperçu rapide des dernières valeurs
    st.subheader("📊 Dernières mesures disponibles")
    if not df.empty:
        latest = (
            df.groupby(["pollutant", "segment"])
            .apply(lambda x: x.nlargest(1, "time"))
            .reset_index(drop=True)
        )
        for pollutant in ["NO2", "PM10", "PM25"]:
            sub = latest[latest["pollutant"] == pollutant].set_index("segment")["value"]
            if not sub.empty:
                st.markdown(f"**{pollutant}** (dernière heure connue)")
                cols = st.columns(len(SEGMENTS))
                for i, seg in enumerate(SEGMENTS):
                    val = sub.get(seg, np.nan)
                    seuil = SEUILS[pollutant]
                    delta = f"{val - seuil:+.1f}" if not np.isnan(val) else None
                    cols[i].metric(
                        seg,
                        f"{val:.0f} µg/m³" if not np.isnan(val) else "N/A",
                        delta=delta,
                        delta_color="inverse",
                    )

    st.divider()
    st.subheader("👥 Équipe projet")
    cols = st.columns(4)
    roles = [
        ("👷 Data Engineer",  "Pipeline ETL, bases de données"),
        ("🤖 Data Scientist", "Modèles prédictifs XGBoost"),
        ("💻 Dev Web",        "Dashboard Streamlit & API"),
        ("🌿 Green IT",       "Données environnementales"),
    ]
    for col, (titre, desc) in zip(cols, roles):
        with col:
            st.info(f"**{titre}**\n{desc}")


# ---------------------------------------------------------------------------
# PAGE 2 : Prédictions
# ---------------------------------------------------------------------------

def page_predictions(df: pd.DataFrame):
    st.title("📈 Prédictions de pollution")
    st.caption("Prédiction horaire basée sur le modèle XGBoost entraîné sur 2024-2025.")

    if df.empty:
        st.error("Données historiques indisponibles.")
        return

    col_sel, col_opts = st.columns([1, 2])

    with col_sel:
        pollutant = st.selectbox("Polluant", ["NO2", "PM10", "PM25"])
        segment   = st.selectbox("Segment du périphérique", SEGMENTS, index=2)
        horizon   = st.slider("Horizon de prédiction (heures)", 1, 48, 24)

    model, model_version = load_model(pollutant)

    with col_opts:
        st.markdown(f"**Modèle chargé :** XGBoost {model_version or 'non disponible'}")
        if model is None:
            st.warning(
                "Aucun modèle trouvé. Lance : `python main.py --meteo data/meteo_2025.csv`"
            )

    # Filtrer les données pour le segment sélectionné
    seg_data = df[
        (df["segment"] == segment) & (df["pollutant"] == pollutant)
    ].sort_values("time")

    last_time = seg_data["time"].max()

    # Graphique historique 7 jours + prédictions récursives (si modèle dispo)
    st.subheader(f"📊 {pollutant} — Segment {segment}")

    hist_7j = seg_data[seg_data["time"] >= last_time - pd.Timedelta(days=7)].copy()
    hist_7j["type"] = "Historique"

    if model is not None:
        # Prédictions récursives simples (basées sur lags disponibles)
        preds = []
        recent = seg_data.set_index("time")["value"].tail(200)

        for h in range(1, horizon + 1):
            future_t = last_time + pd.Timedelta(hours=h)
            # Construire features minimales
            lag_vals = {}
            for lag in [1, 2, 3, 6, 24, 168]:
                past_t = future_t - pd.Timedelta(hours=lag)
                if past_t in recent.index:
                    lag_vals[f"lag_{lag}h"] = recent[past_t]
                elif preds and lag <= len(preds):
                    lag_vals[f"lag_{lag}h"] = preds[-lag]["value"]
                else:
                    lag_vals[f"lag_{lag}h"] = recent.mean()

            row = {
                "hour": future_t.hour, "day_of_week": future_t.dayofweek,
                "month": future_t.month, "is_weekend": int(future_t.dayofweek >= 5),
                "is_peak_hour": int(future_t.hour in [7, 8, 9, 17, 18, 19]),
                "is_holiday": 0,
                "hour_sin": np.sin(2 * np.pi * future_t.hour / 24),
                "hour_cos": np.cos(2 * np.pi * future_t.hour / 24),
                "dow_sin":  np.sin(2 * np.pi * future_t.dayofweek / 7),
                "dow_cos":  np.cos(2 * np.pi * future_t.dayofweek / 7),
                "month_sin": np.sin(2 * np.pi * future_t.month / 12),
                "month_cos": np.cos(2 * np.pi * future_t.month / 12),
                **lag_vals,
            }
            # Rolling
            past_24 = list(recent.tail(24).values) + [p["value"] for p in preds]
            row["rolling_mean_3h"]  = float(np.mean(past_24[-3:]))  if past_24 else 0
            row["rolling_mean_24h"] = float(np.mean(past_24[-24:])) if past_24 else 0
            row["rolling_std_24h"]  = float(np.std(past_24[-24:]))  if len(past_24) >= 2 else 0

            # Segment one-hot
            for seg in SEGMENTS:
                row[f"seg_{seg}"] = int(seg == segment)

            X = pd.DataFrame([row]).reindex(columns=model.feature_names_, fill_value=np.nan)
            val = max(0.0, float(model.predict(X)[0]))
            preds.append({"time": future_t, "value": val})
            # Mettre à jour recent
            recent[future_t] = val

        preds_df = pd.DataFrame(preds)
        preds_df["type"] = "Prédiction"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_7j["time"], y=hist_7j["value"],
            name="Historique", line=dict(color="#264653", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=preds_df["time"], y=preds_df["value"],
            name="Prédiction", line=dict(color="#E76F51", width=2, dash="dash"),
        ))
        seuil = SEUILS[pollutant]
        fig.add_hline(y=seuil, line_dash="dot", line_color="red",
                      annotation_text=f"Seuil légal {seuil} µg/m³")
        fig.update_layout(
            xaxis_title="Date / Heure",
            yaxis_title=f"{pollutant} (µg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPIs prédictions
        st.subheader("🎯 Alertes prédites")
        max_pred = preds_df["value"].max()
        n_alerts = (preds_df["value"] > SEUILS[pollutant]).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Concentration max prédite", f"{max_pred:.1f} µg/m³",
                  delta=f"{max_pred - SEUILS[pollutant]:+.1f} vs seuil",
                  delta_color="inverse")
        c2.metric("Heures en alerte", f"{n_alerts} / {horizon}")
        c3.metric("Statut actuel",
                  "⚠️ ALERTE" if hist_7j["value"].iloc[-1] > SEUILS[pollutant] else "✅ OK")

    else:
        # Juste l'historique si pas de modèle
        fig = px.line(hist_7j, x="time", y="value", title=f"{pollutant} — {segment}")
        st.plotly_chart(fig, use_container_width=True)

    # Tableau des dernières prédictions
    if model is not None and preds:
        with st.expander("📋 Détail des prédictions horaires"):
            preds_display = pd.DataFrame(preds).copy()
            preds_display["time"] = preds_display["time"].dt.strftime("%Y-%m-%d %H:%M")
            preds_display["value"] = preds_display["value"].round(2)
            preds_display["alerte"] = preds_display["value"].apply(
                lambda v: "⚠️ OUI" if v > SEUILS[pollutant] else "✅ NON"
            )
            preds_display.columns = ["Datetime", f"{pollutant} (µg/m³)", "Alerte"]
            st.dataframe(preds_display, use_container_width=True)

        # Export
        csv = preds_df.to_csv(index=False)
        st.download_button(
            "📥 Télécharger prédictions (CSV)",
            data=csv,
            file_name=f"predictions_{pollutant}_{segment}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# PAGE 3 : Analyses
# ---------------------------------------------------------------------------

def page_analyses(df: pd.DataFrame):
    st.title("🔬 Analyses exploratoires")

    if df.empty:
        st.error("Données indisponibles.")
        return

    pollutant = st.selectbox("Polluant à analyser", ["NO2", "PM10", "PM25"])
    sub = df[df["pollutant"] == pollutant]

    col1, col2 = st.columns(2)

    with col1:
        # Profil horaire moyen par segment
        st.subheader("Profil horaire moyen")
        hourly = sub.groupby(["hour", "segment"])["value"].mean().reset_index()
        fig = px.line(hourly, x="hour", y="value", color="segment",
                      title=f"{pollutant} — Moyenne horaire par segment",
                      labels={"value": "µg/m³", "hour": "Heure"})
        fig.add_hline(y=SEUILS[pollutant], line_dash="dot", line_color="red",
                      annotation_text="Seuil légal")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Boxplot par jour de semaine
        st.subheader("Distribution par jour de semaine")
        sub_dow = sub.copy()
        sub_dow["day_of_week"] = sub_dow["time"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig2 = px.box(sub_dow, x="day_of_week", y="value", color="segment",
                      category_orders={"day_of_week": day_order},
                      title=f"{pollutant} — Distribution par jour",
                      labels={"value": "µg/m³"})
        st.plotly_chart(fig2, use_container_width=True)

    # Carte des segments (tableau de bord)
    st.subheader("📊 Concentration moyenne par segment (période complète)")
    seg_mean = sub.groupby("segment")["value"].agg(["mean", "max", "std"]).round(2)
    seg_mean.columns = ["Moyenne µg/m³", "Max µg/m³", "Écart-type"]
    seg_mean["Seuil légal"] = SEUILS[pollutant]
    seg_mean["% au-dessus seuil"] = (
        sub.groupby("segment")["value"].apply(lambda x: (x > SEUILS[pollutant]).mean() * 100)
    ).round(1)
    st.dataframe(
        seg_mean.style.background_gradient(subset=["Moyenne µg/m³"], cmap="Reds"),
        use_container_width=True,
    )

    # Tendance mensuelle
    st.subheader("📅 Évolution mensuelle")
    sub_monthly = sub.copy()
    sub_monthly["month"] = sub_monthly["time"].dt.to_period("M").astype(str)
    monthly_avg = sub_monthly.groupby(["month", "segment"])["value"].mean().reset_index()
    fig3 = px.line(monthly_avg, x="month", y="value", color="segment",
                   title=f"{pollutant} — Moyenne mensuelle",
                   labels={"value": "µg/m³", "month": "Mois"})
    fig3.add_hline(y=SEUILS[pollutant], line_dash="dot", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap heure × jour
    st.subheader("🔥 Heatmap — Heure × Jour de semaine")
    seg_choice = st.selectbox("Segment pour la heatmap", SEGMENTS, index=2, key="seg_heat")
    sub_heat = sub[sub["segment"] == seg_choice].copy()
    sub_heat["dow"]  = sub_heat["time"].dt.dayofweek
    sub_heat["hour"] = sub_heat["time"].dt.hour
    heat = sub_heat.groupby(["dow", "hour"])["value"].mean().unstack()
    heat.index = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    fig4 = px.imshow(heat, title=f"{pollutant} — {seg_choice} — heure × jour",
                     color_continuous_scale="Reds", aspect="auto",
                     labels={"color": "µg/m³"})
    st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 4 : Modèle
# ---------------------------------------------------------------------------

def page_modele():
    st.title("⚙️ Internals du modèle")

    pollutant = st.selectbox("Polluant", ["NO2", "PM10", "PM25"])
    model, model_version = load_model(pollutant)

    if model is None:
        st.warning(
            "Aucun modèle entraîné trouvé.\n\n"
            "Lance : `python main.py --meteo data/meteo_2025.csv --pollutant NO2`"
        )
        return

    st.success(f"✅ Modèle XGBoost {model_version} chargé | {len(model.feature_names_)} features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔬 Importance des features")
        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(top=20)
        else:
            imp = pd.DataFrame({
                "feature": model.feature_names_,
                "importance": model.model.feature_importances_,
            }).sort_values("importance", ascending=False).head(20)

        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="Top 20 features par importance",
                     labels={"importance": "Importance", "feature": "Feature"},
                     color="importance", color_continuous_scale="Teal")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Benchmark des modèles")
        metrics = load_metrics(pollutant, model_version or "v2")
        if metrics:
            df_m = pd.DataFrame(metrics)
            cols_show = ["modèle", "set", "MAE", "RMSE", "MAPE",
                         "recall_exceed", "precision_exceed"]
            df_m_show = df_m[[c for c in cols_show if c in df_m.columns]]
            df_m_show = df_m_show[df_m_show["set"] == "test"].drop(columns=["set"])
            st.dataframe(
                df_m_show.style.background_gradient(subset=["MAE"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

            # Graphique benchmark
            test_metrics = df_m[df_m["set"] == "test"]
            if len(test_metrics) > 0:
                fig2 = px.bar(test_metrics, x="modèle", y="MAE",
                              color="modèle", title="MAE sur set de test par modèle")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Métriques non disponibles. Entraîne d'abord le modèle.")

    # Hyperparamètres
    st.subheader("⚙️ Hyperparamètres")
    if hasattr(model, "model") and model.model is not None:
        params = model.model.get_params()
        params_df = pd.DataFrame.from_dict(params, orient="index", columns=["Valeur"])
        st.dataframe(params_df, use_container_width=True)

    # Infos modèle
    st.subheader("ℹ️ Informations")
    info = {
        "Polluant cible": model.pollutant if hasattr(model, "pollutant") else pollutant,
        "Version": model_version,
        "Nb features": len(model.feature_names_),
        "Features météo incluses": "Oui" if any("TEMPERATURE" in f for f in model.feature_names_) else "Non",
        "Seuil légal (alerte)": f"{SEUILS.get(pollutant, 40)} µg/m³",
    }
    for k, v in info.items():
        st.write(f"**{k}** : {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_pollution_history()
    page = sidebar()

    if "Accueil" in page:
        page_accueil(df)
    elif "Prédictions" in page:
        page_predictions(df)
    elif "Analyses" in page:
        page_analyses(df)
    elif "Modèle" in page:
        page_modele()


if __name__ == "__main__":
    main()
