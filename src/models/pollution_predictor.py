"""
Prédicteur de pollution sur le périphérique parisien.

Encapsule un modèle XGBoost entraîné sur le dataset enrichi
(features cycliques + lags + rolling + jours fériés + météo optionnelle).

Utilisation :
    from src.models.pollution_predictor import PollutionPredictor
    model = PollutionPredictor()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = model.evaluate(X_val, y_val)
    model.save("models/pollution_xgb_no2.pkl")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Colonnes utilisées comme features (toutes celles qui ne sont ni la cible ni des
# identifiants temporels bruts). Ajustées dans .fit() en filtrant ce qui existe.
DEFAULT_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
]

# Encodage des segments (catégories) - one-hot
SEGMENT_COL = "segment"


@dataclass
class PollutionPredictor:
    """XGBoost regressor pour la prédiction horaire de pollution."""

    pollutant: str = "NO2"
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42

    model: object | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne les features disponibles + one-hot des segments."""
        # Features numériques disponibles
        feats = [c for c in DEFAULT_FEATURES if c in df.columns]
        X = df[feats].copy()

        # One-hot des segments
        if SEGMENT_COL in df.columns:
            seg_dummies = pd.get_dummies(df[SEGMENT_COL], prefix="seg", dtype=int)
            X = pd.concat([X.reset_index(drop=True), seg_dummies.reset_index(drop=True)], axis=1)

        return X

    def fit(self, df_train: pd.DataFrame, target_col: str = "value") -> "PollutionPredictor":
        """Entraîne le modèle. df_train doit déjà être filtré sur le polluant."""
        df_train = df_train.dropna(subset=[target_col]).copy()
        # On retire les lignes où les lags sont NaN (début de série)
        lag_cols = [c for c in df_train.columns if c.startswith("lag_") or c.startswith("rolling_")]
        if lag_cols:
            df_train = df_train.dropna(subset=lag_cols)

        X = self._build_X(df_train)
        y = df_train[target_col].values
        self.feature_names_ = X.columns.tolist()

        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost n'est pas installé. Lance : pip install xgboost"
            ) from e

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            tree_method="hist",
            n_jobs=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelle .fit() d'abord.")
        X = self._build_X(df)
        # Réaligner les colonnes sur celles vues à l'entraînement
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        return self.model.predict(X)

    def evaluate(
        self, df: pd.DataFrame, target_col: str = "value", threshold: float | None = None
    ) -> dict:
        """Calcule MAE, RMSE, MAPE et taux de bonne détection des dépassements."""
        df = df.dropna(subset=[target_col]).copy()
        lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_")]
        if lag_cols:
            df = df.dropna(subset=lag_cols)

        y_true = df[target_col].values
        y_pred = self.predict(df)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # MAPE robuste : ignore les valeurs cibles trop faibles
        mask = np.abs(y_true) > 1
        mape = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.any()
            else np.nan
        )

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": len(y_true)}

        if threshold is not None:
            true_exceed = y_true > threshold
            pred_exceed = y_pred > threshold
            tp = int(((true_exceed) & (pred_exceed)).sum())
            fp = int(((~true_exceed) & (pred_exceed)).sum())
            fn = int(((true_exceed) & (~pred_exceed)).sum())
            tn = int(((~true_exceed) & (~pred_exceed)).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            metrics.update(
                {
                    "threshold": threshold,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "recall_exceed": recall,
                    "precision_exceed": precision,
                }
            )

        return metrics

    def feature_importance(self, top: int = 15) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Modèle non entraîné.")
        imp = pd.DataFrame(
            {"feature": self.feature_names_, "importance": self.model.feature_importances_}
        )
        return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PollutionPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


# =============================================================================
# Baselines pour benchmark
# =============================================================================


def baseline_seasonal_naive(
    df: pd.DataFrame,
    target_col: str = "value",
    lag_col: str = "lag_168h",
) -> np.ndarray:
    """Baseline naïve saisonnière : prédit la valeur de la même heure 7j avant.
    Lag par défaut = 168h (1 semaine)."""
    return df[lag_col].values


def baseline_persistence(df: pd.DataFrame, lag_col: str = "lag_1h") -> np.ndarray:
    """Baseline persistante : prédit la valeur de l'heure précédente."""
    return df[lag_col].values


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float | None = None
) -> dict:
    """Version standalone des métriques, pour benchmark des baselines."""
    mask_finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask_finite], y_pred[mask_finite]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any()
        else np.nan
    )
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": len(y_true)}

    if threshold is not None:
        true_exceed = y_true > threshold
        pred_exceed = y_pred > threshold
        tp = int(((true_exceed) & (pred_exceed)).sum())
        fp = int(((~true_exceed) & (pred_exceed)).sum())
        fn = int(((true_exceed) & (~pred_exceed)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        metrics.update(
            {
                "threshold": threshold,
                "recall_exceed": recall,
                "precision_exceed": precision,
            }
        )
    return metrics
