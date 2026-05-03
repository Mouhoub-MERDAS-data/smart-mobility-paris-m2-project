"""
Module de prédiction sur nouvelles données.

Charge un modèle entraîné et produit des prédictions pour :
  - Un segment + datetime précis
  - Un segment sur les N prochaines heures (forecast)
  - Tous les segments pour un instant donné (snapshot)

Usage :
    from src.models.predict import Predictor
    pred = Predictor.from_model_dir("models/", pollutant="NO2", version="v2")
    result = pred.forecast(segment="Berc-Ital", from_datetime="2025-06-01 08:00", horizon=24)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
SEUILS       = {"NO2": 40.0, "PM10": 40.0, "PM25": 25.0}

SEGMENTS = [
    "Chap-Bagn", "Bagn-Berc", "Berc-Ital", "Ital-A6a",
    "A6a-Sevr",  "Sevr-Aute", "Aute-Mail", "Mail-Chap",
]


class Predictor:
    """
    Interface haut niveau pour la prédiction de pollution.

    Charge le modèle XGBoost entraîné + les données historiques nécessaires
    pour construire les features (lags) et retourne des prédictions prêtes
    à afficher dans le Streamlit.
    """

    def __init__(
        self,
        model,
        history: pd.DataFrame,
        pollutant: str = "NO2",
        meteo: pd.DataFrame | None = None,
    ):
        self.model     = model
        self.history   = history.copy()
        self.pollutant = pollutant
        self.meteo     = meteo
        self.threshold = SEUILS.get(pollutant, 40.0)
        self._fe       = None

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path = MODELS_DIR,
        pollutant: str = "NO2",
        version: str = "v2",
        history: pd.DataFrame | None = None,
        meteo_path: str | Path | None = None,
    ) -> "Predictor":
        """
        Charge le modèle depuis le dossier models/.

        Args:
            model_dir  : dossier contenant les .pkl
            pollutant  : polluant cible
            version    : 'v1' (sans météo) ou 'v2' (avec météo)
            history    : DataFrame avec l'historique de pollution (format long)
            meteo_path : chemin vers le CSV météo daily (pour v2)
        """
        model_dir = Path(model_dir)
        model_path = model_dir / f"xgboost_{pollutant.lower()}_{version}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                f"Lance d'abord : python -m src.models.train --pollutant {pollutant}"
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        log.info(f"Modèle chargé : {model_path.name}")

        # Charger l'historique si non fourni
        if history is None:
            from src.data.build_dataset import build_full_dataset
            meteo = meteo_path if version == "v2" else None
            history = build_full_dataset(
                pollutants=[pollutant],
                meteo_path=meteo,
            )
            history = history[history["pollutant"] == pollutant]

        meteo_df = None
        if meteo_path is not None and version == "v2":
            from src.data.build_dataset import load_and_interpolate_meteo
            meteo_df = load_and_interpolate_meteo(meteo_path)

        return cls(model=model, history=history, pollutant=pollutant, meteo=meteo_df)

    # ------------------------------------------------------------------
    # Préparation des features pour un point donné
    # ------------------------------------------------------------------

    def _build_features_for_datetime(
        self,
        segment: str,
        dt: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Construit le vecteur de features pour un segment + instant.
        Utilise l'historique réel pour les lags.
        """
        from src.features.feature_engineering import FeatureEngineer, METEO_FEATURES
        import numpy as np

        # Récupérer l'historique de ce segment
        seg_hist = (
            self.history[self.history["segment"] == segment]
            .sort_values("time")
            .set_index("time")
        )

        # Features temporelles
        row = {
            "time":        dt,
            "segment":     segment,
            "pollutant":   self.pollutant,
            "hour":        dt.hour,
            "day_of_week": dt.dayofweek,
            "day_name":    dt.day_name(),
            "month":       dt.month,
            "year":        dt.year,
            "is_weekend":  int(dt.dayofweek >= 5),
            "is_peak_hour": int(dt.hour in [7, 8, 9, 17, 18, 19]),
            "is_holiday":  0,  # simplifié
        }

        # Cycliques
        row["hour_sin"]  = np.sin(2 * np.pi * dt.hour / 24)
        row["hour_cos"]  = np.cos(2 * np.pi * dt.hour / 24)
        row["dow_sin"]   = np.sin(2 * np.pi * dt.dayofweek / 7)
        row["dow_cos"]   = np.cos(2 * np.pi * dt.dayofweek / 7)
        row["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

        # Lags depuis l'historique réel
        for lag in [1, 2, 3, 6, 24, 168]:
            target_time = dt - pd.Timedelta(hours=lag)
            if target_time in seg_hist.index:
                row[f"lag_{lag}h"] = float(seg_hist.loc[target_time, "value"])
            else:
                row[f"lag_{lag}h"] = np.nan

        # Rolling means
        past_vals = seg_hist["value"][seg_hist.index < dt].tail(24)
        row["rolling_mean_3h"]  = float(past_vals.tail(3).mean())  if len(past_vals) >= 1 else np.nan
        row["rolling_mean_24h"] = float(past_vals.tail(24).mean()) if len(past_vals) >= 1 else np.nan
        row["rolling_std_24h"]  = float(past_vals.tail(24).std())  if len(past_vals) >= 2 else np.nan

        # One-hot segment
        for seg in SEGMENTS:
            row[f"seg_{seg}"] = int(seg == segment)

        # Météo (si disponible)
        if self.meteo is not None:
            meteo_cols = [c for c in METEO_FEATURES if c in self.meteo.columns]
            meteo_row = self.meteo[self.meteo["time"] == dt]
            for col in meteo_cols:
                row[col] = float(meteo_row[col].values[0]) if len(meteo_row) > 0 else np.nan

        df_row = pd.DataFrame([row])
        return df_row

    # ------------------------------------------------------------------
    # Prédiction ponctuelle
    # ------------------------------------------------------------------

    def predict_single(
        self, segment: str, dt: str | pd.Timestamp
    ) -> dict:
        """
        Prédit la concentration pour un segment + instant.

        Returns:
            dict avec 'prediction', 'threshold', 'is_alert', 'segment', 'datetime'
        """
        dt = pd.Timestamp(dt)
        X = self._build_features_for_datetime(segment, dt)
        X_aligned = X.reindex(columns=self.model.feature_names_, fill_value=np.nan)
        pred = float(self.model.predict(X_aligned)[0])
        pred = max(0.0, pred)

        return {
            "segment":    segment,
            "datetime":   dt,
            "prediction": round(pred, 2),
            "threshold":  self.threshold,
            "is_alert":   pred > self.threshold,
            "pollutant":  self.pollutant,
        }

    # ------------------------------------------------------------------
    # Forecast : N heures à l'avance
    # ------------------------------------------------------------------

    def forecast(
        self,
        segment: str,
        from_datetime: str | pd.Timestamp,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """
        Prédit la concentration sur 'horizon' heures à partir de from_datetime.

        Utilise l'historique réel jusqu'à from_datetime, puis des prédictions
        récursives pour les heures suivantes (recursive forecasting).

        Returns:
            DataFrame avec colonnes : time, prediction, is_alert, is_forecast
        """
        from_dt = pd.Timestamp(from_datetime)

        # Récupérer l'historique récent (pour le graphique)
        seg_hist = (
            self.history[
                (self.history["segment"] == segment) &
                (self.history["time"] >= from_dt - pd.Timedelta(hours=48)) &
                (self.history["time"] < from_dt)
            ]
            .sort_values("time")[["time", "value"]]
            .copy()
        )
        seg_hist["is_forecast"] = False
        seg_hist["is_alert"] = seg_hist["value"] > self.threshold
        seg_hist = seg_hist.rename(columns={"value": "prediction"})

        # Prédictions futures (récursives)
        preds = []
        working_history = self.history.copy()

        for h in range(1, horizon + 1):
            target_dt = from_dt + pd.Timedelta(hours=h)
            X = self._build_features_for_datetime(segment, target_dt)
            X_aligned = X.reindex(columns=self.model.feature_names_, fill_value=np.nan)
            pred_val = max(0.0, float(self.model.predict(X_aligned)[0]))

            preds.append({
                "time":        target_dt,
                "prediction":  round(pred_val, 2),
                "is_forecast": True,
                "is_alert":    pred_val > self.threshold,
            })

            # Ajouter la prédiction à l'historique pour les lags suivants
            new_row = pd.DataFrame([{
                "time": target_dt, "segment": segment,
                "pollutant": self.pollutant, "value": pred_val,
                **{c: np.nan for c in working_history.columns
                   if c not in ["time", "segment", "pollutant", "value"]},
            }])
            working_history = pd.concat([working_history, new_row], ignore_index=True)
            self.history = working_history  # mise à jour temporaire

        forecast_df = pd.DataFrame(preds)
        result = pd.concat([seg_hist, forecast_df], ignore_index=True)
        return result.sort_values("time").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Snapshot : tous les segments pour un instant
    # ------------------------------------------------------------------

    def snapshot(self, dt: str | pd.Timestamp) -> pd.DataFrame:
        """
        Prédit la concentration sur TOUS les segments pour un instant donné.

        Returns:
            DataFrame avec colonnes : segment, prediction, is_alert
        """
        dt = pd.Timestamp(dt)
        rows = []
        for seg in SEGMENTS:
            r = self.predict_single(seg, dt)
            rows.append(r)
        return pd.DataFrame(rows)
