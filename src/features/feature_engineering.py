"""
Feature engineering complet pour la prédiction de pollution.

Transforme le dataset brut (pollution + météo + contexte) en features
prêtes à l'entraînement XGBoost.

Usage :
    from src.features.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df_raw)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SEGMENTS = [
    "Chap-Bagn", "Bagn-Berc", "Berc-Ital", "Ital-A6a",
    "A6a-Sevr",  "Sevr-Aute", "Aute-Mail", "Mail-Chap",
]

# Features météo disponibles si le fichier est fourni
METEO_FEATURES = [
    "MAX_TEMPERATURE_C", "MIN_TEMPERATURE_C", "TEMP_RANGE_C",
    "HUMIDITY_MAX_PERCENT", "PRECIP_TOTAL_DAY_MM",
    "PRESSURE_MAX_MB", "CLOUDCOVER_AVG_PERCENT",
    "WINDSPEED_MAX_KMH", "VISIBILITY_AVG_KM",
    "UV_INDEX", "SUNHOUR", "TOTAL_SNOW_MM",
    "IS_COLD", "IS_HOT", "IS_RAINY", "IS_SNOWY", "IS_OVERCAST",
    "TEMPERATURE_MORNING_C_6H", "TEMPERATURE_NOON_C_12H",
    "TEMPERATURE_EVENING_C_18H", "TEMPERATURE_NIGHT_C_3H",
]

# Features toujours présentes (sans météo)
BASE_FEATURES = [
    # Cycliques
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    # Contexte
    "hour", "day_of_week", "month",
    "is_weekend", "is_peak_hour", "is_holiday",
    # Lags pollution
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_24h", "lag_168h",
    # Rolling
    "rolling_mean_3h", "rolling_mean_24h", "rolling_std_24h",
    # Segment (one-hot)
]


@dataclass
class FeatureEngineer:
    """
    Transforme un DataFrame pollution+météo en features XGBoost.

    Attributes:
        lags         : fenêtres lag en heures
        rolling_wins : fenêtres rolling en heures
        with_meteo   : inclure les features météo (si disponibles)
    """

    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 24, 168])
    rolling_wins: list[int] = field(default_factory=lambda: [3, 24])
    with_meteo: bool = True
    _fitted: bool = field(default=False, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Cycliques
    # ------------------------------------------------------------------

    @staticmethod
    def _add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
        return df

    # ------------------------------------------------------------------
    # Lags  (shift par groupe pollutant × segment)
    # ------------------------------------------------------------------

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["pollutant", "segment", "time"])
        grp = df.groupby(["pollutant", "segment"])["value"]
        for lag in self.lags:
            df[f"lag_{lag}h"] = grp.shift(lag)
        return df

    # ------------------------------------------------------------------
    # Rolling means / std
    # ------------------------------------------------------------------

    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["pollutant", "segment", "time"])
        grp = df.groupby(["pollutant", "segment"])["value"]
        for w in self.rolling_wins:
            df[f"rolling_mean_{w}h"] = (
                grp.shift(1).rolling(w, min_periods=1).mean().reset_index(drop=True)
            )
        # Écart-type 24h (mesure de la variabilité récente)
        df["rolling_std_24h"] = (
            grp.shift(1).rolling(24, min_periods=2).std().reset_index(drop=True)
        )
        return df

    # ------------------------------------------------------------------
    # One-hot segment
    # ------------------------------------------------------------------

    @staticmethod
    def _add_segment_dummies(df: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(df["segment"], prefix="seg", dtype=int)
        return pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, target_col: str = "value") -> pd.DataFrame:
        """
        Applique toutes les transformations et retourne le dataset enrichi.
        Supprime les lignes où les lags critiques (lag_1h, lag_24h) sont NaN.
        """
        log.info("Feature engineering → début")

        df = self._add_cyclical(df)
        df = self._add_lags(df)
        df = self._add_rolling(df)
        df = self._add_segment_dummies(df)

        # Colonnes de features effectives
        seg_cols = [c for c in df.columns if c.startswith("seg_")]
        meteo_cols = [c for c in METEO_FEATURES if c in df.columns] if self.with_meteo else []

        self._feature_names = BASE_FEATURES + seg_cols + meteo_cols
        # Garder seulement les features qui existent réellement dans df
        self._feature_names = [c for c in self._feature_names if c in df.columns]

        # Supprimer les lignes sans lag critique ET sans cible
        required_lags = [c for c in ["lag_1h", "lag_24h"] if c in df.columns]
        drop_cols = required_lags + (["value"] if "value" in df.columns else [])
        df = df.dropna(subset=drop_cols).reset_index(drop=True)

        self._fitted = True
        log.info(
            f"Feature engineering terminé : {len(df):,} lignes × "
            f"{len(self._feature_names)} features | météo={'oui' if meteo_cols else 'non'}"
        )
        return df

    def get_X_y(
        self, df: pd.DataFrame, target_col: str = "value"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Retourne X (features) et y (cible) prêts pour XGBoost."""
        if not self._fitted:
            raise RuntimeError("Appelle fit_transform() d'abord.")
        X = df[self._feature_names].copy()
        y = df[target_col]
        return X, y

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def has_meteo(self, df: pd.DataFrame) -> bool:
        """Vérifie si des colonnes météo sont présentes dans le df."""
        return any(c in df.columns for c in METEO_FEATURES)


# ------------------------------------------------------------------
# Split temporel
# ------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2024-12-31 23:00:00",
    val_end: str = "2025-09-30 23:00:00",
    time_col: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporel strict — aucun overlap possible.

    Par défaut :
        Train : 2024 complet
        Val   : 2025 janv → sept
        Test  : 2025 oct → 2026 mars
    """
    train = df[df[time_col] <= train_end].copy()
    val   = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test  = df[df[time_col] > val_end].copy()

    log.info(
        f"Split | Train: {len(train):,} ({train[time_col].min().date()}→{train[time_col].max().date()}) "
        f"| Val: {len(val):,} ({val[time_col].min().date()}→{val[time_col].max().date()}) "
        f"| Test: {len(test):,} ({test[time_col].min().date()}→{test[time_col].max().date()})"
    )
    return train, val, test


def temporal_split_v2(
    df: pd.DataFrame,
    train_end: str = "2025-09-30 23:00:00",
    val_end: str = "2025-12-31 23:00:00",
    time_col: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split pour le modèle V2 (avec météo 2025 complète).

    Train : 2024 + 2025 janv → sept
    Val   : 2025 oct → déc
    Test  : 2026 (toutes années restantes)

    Note : les données 2024 n'ont pas de météo → les features météo
    seront NaN pour cette période. XGBoost gère nativement les NaN
    (tree_method='hist'), donc l'entraînement reste valide mais
    le modèle apprendra les features météo principalement sur 2025.
    """
    train = df[df[time_col] <= train_end].copy()
    val   = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test  = df[df[time_col] > val_end].copy()

    log.info(
        f"Split V2 | Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}"
    )
    return train, val, test
