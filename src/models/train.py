"""
Entraînement, évaluation et comparaison des modèles de prédiction de pollution.

Modèles disponibles :
  - baseline_persistence   : valeur h-1 (la plus simple)
  - baseline_seasonal      : valeur J-7 même heure
  - baseline_ridge         : régression Ridge (ML simple)
  - xgboost_v1             : XGBoost sans météo
  - xgboost_v2             : XGBoost avec météo 2025

Usage :
    python -m src.models.train --meteo data/meteo_2025.csv
    python -m src.models.train --help
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEUILS = {"NO2": 40.0, "PM10": 40.0, "PM25": 25.0}

# ------------------------------------------------------------------
# Métriques
# ------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """MAE, RMSE, MAPE + détection de dépassements si threshold fourni."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]

    mae  = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    pos_mask = y_t > 1
    mape = float(np.mean(np.abs((y_t[pos_mask] - y_p[pos_mask]) / y_t[pos_mask])) * 100) \
        if pos_mask.any() else float("nan")

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": int(len(y_t))}

    if threshold is not None:
        true_exc = y_t > threshold
        pred_exc = y_p > threshold
        tp = int((true_exc & pred_exc).sum())
        fp = int((~true_exc & pred_exc).sum())
        fn = int((true_exc & ~pred_exc).sum())
        tn = int((~true_exc & ~pred_exc).sum())
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = 2 * recall * precision / (recall + precision) \
            if (recall + precision) > 0 else float("nan")
        metrics.update({
            "threshold": threshold,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall_exceed": recall,
            "precision_exceed": precision,
            "f1_exceed": f1,
        })

    return metrics


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------

def predict_persistence(df: pd.DataFrame) -> np.ndarray:
    """Prédit la valeur de h-1."""
    if "lag_1h" not in df.columns:
        raise ValueError("Colonne lag_1h manquante. Lance le feature engineering d'abord.")
    return df["lag_1h"].values


def predict_seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Prédit la valeur de la même heure 7j avant."""
    if "lag_168h" not in df.columns:
        raise ValueError("Colonne lag_168h manquante.")
    return df["lag_168h"].values


# ------------------------------------------------------------------
# Ridge Regression (baseline ML)
# ------------------------------------------------------------------

class RidgeBaseline:
    """Wrapper Ridge avec interface identique à PollutionPredictor."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_names_: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeBaseline":
        self.feature_names_ = X_train.columns.tolist()
        X = X_train.fillna(0)
        self.model.fit(X, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names_, fill_value=0).fillna(0)
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# XGBoost Predictor
# ------------------------------------------------------------------

@dataclass
class PollutionPredictor:
    """
    XGBoost regressor pour la prédiction horaire de pollution.

    Supporte à la fois le mode sans météo (v1) et avec météo (v2).
    Les NaN dans les features météo sont gérés nativement par XGBoost
    (tree_method='hist') — aucun fillna nécessaire.
    """

    pollutant: str = "NO2"
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    model_version: str = "v1"

    model: object = field(default=None, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    train_metrics_: dict = field(default_factory=dict, init=False)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "PollutionPredictor":
        """
        Entraîne le modèle. XGBoost gère les NaN nativement (tree_method='hist').
        Si X_val est fourni, utilise early stopping.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Installe xgboost : pip install xgboost")

        self.feature_names_ = X_train.columns.tolist()

        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            tree_method="hist",
            n_jobs=-1,
            enable_categorical=False,
        )

        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = 50
            self.model = XGBRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val.reindex(columns=self.feature_names_), y_val)],
                verbose=100,
            )
            log.info(f"Best iteration : {self.model.best_iteration}")
        else:
            self.model = XGBRegressor(**params)
            self.model.fit(X_train, y_train)

        log.info(
            f"XGBoost {self.model_version} entraîné | "
            f"{len(self.feature_names_)} features | "
            f"polluant={self.pollutant}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelle .fit() d'abord.")
        X_aligned = X.reindex(columns=self.feature_names_)
        return self.model.predict(X_aligned)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float | None = None,
    ) -> dict:
        preds = self.predict(X)
        metrics = compute_metrics(y.values, preds, threshold=threshold)
        return metrics

    def feature_importance(self, top: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Modèle non entraîné.")
        imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model.feature_importances_,
        })
        return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Modèle sauvegardé → {path} ({path.stat().st_size / 1024:.1f} Ko)")

    @classmethod
    def load(cls, path: str | Path) -> "PollutionPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# Benchmark complet
# ------------------------------------------------------------------

def run_benchmark(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pollutant: str = "NO2",
    model_version: str = "v1",
    save_model: bool = True,
) -> tuple[pd.DataFrame, PollutionPredictor]:
    """
    Lance tous les modèles et retourne un tableau comparatif + le meilleur modèle.

    Returns:
        (DataFrame résultats, PollutionPredictor entraîné)
    """
    threshold = SEUILS.get(pollutant)
    results = []

    # ---- Baseline persistance ----
    try:
        y_pred_p = predict_persistence(df_val)
        m = compute_metrics(y_val.values, y_pred_p, threshold)
        results.append({"modèle": "Persistance (h-1)", "set": "val", **m})
        y_pred_p_test = predict_persistence(df_test)
        m = compute_metrics(y_test.values, y_pred_p_test, threshold)
        results.append({"modèle": "Persistance (h-1)", "set": "test", **m})
        log.info("Baseline persistance : OK")
    except Exception as e:
        log.warning(f"Baseline persistance : {e}")

    # ---- Baseline saisonnier ----
    try:
        y_pred_s = predict_seasonal_naive(df_val)
        m = compute_metrics(y_val.values, y_pred_s, threshold)
        results.append({"modèle": "Naïf saisonnier (J-7)", "set": "val", **m})
        y_pred_s_test = predict_seasonal_naive(df_test)
        m = compute_metrics(y_test.values, y_pred_s_test, threshold)
        results.append({"modèle": "Naïf saisonnier (J-7)", "set": "test", **m})
        log.info("Baseline saisonnier : OK")
    except Exception as e:
        log.warning(f"Baseline saisonnier : {e}")

    # ---- Ridge ----
    try:
        ridge = RidgeBaseline()
        ridge.fit(X_train.fillna(0), y_train)
        y_pred_r = ridge.predict(X_val)
        m = compute_metrics(y_val.values, y_pred_r, threshold)
        results.append({"modèle": "Ridge regression", "set": "val", **m})
        y_pred_r_test = ridge.predict(X_test)
        m = compute_metrics(y_test.values, y_pred_r_test, threshold)
        results.append({"modèle": "Ridge regression", "set": "test", **m})
        if save_model:
            ridge.save(MODELS_DIR / f"ridge_{pollutant.lower()}_{model_version}.pkl")
        log.info("Ridge : OK")
    except Exception as e:
        log.warning(f"Ridge : {e}")

    # ---- XGBoost ----
    xgb = PollutionPredictor(pollutant=pollutant, model_version=model_version)
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_pred_x = xgb.predict(X_val)
    m = compute_metrics(y_val.values, y_pred_x, threshold)
    results.append({"modèle": f"XGBoost {model_version}", "set": "val", **m})

    y_pred_x_test = xgb.predict(X_test)
    m = compute_metrics(y_test.values, y_pred_x_test, threshold)
    results.append({"modèle": f"XGBoost {model_version}", "set": "test", **m})
    log.info(f"XGBoost {model_version} : OK | MAE val = {results[-2]['MAE']:.3f}")

    if save_model:
        xgb.save(MODELS_DIR / f"xgboost_{pollutant.lower()}_{model_version}.pkl")
        # Sauvegarder aussi les métriques
        metrics_path = MODELS_DIR / f"metrics_{pollutant.lower()}_{model_version}.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Métriques sauvegardées → {metrics_path}")

    df_results = pd.DataFrame(results)
    return df_results, xgb


# ------------------------------------------------------------------
# Entrée CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Entraîne les modèles de pollution")
    parser.add_argument("--meteo",     default=None,  help="Chemin CSV météo daily")
    parser.add_argument("--pollutant", default="NO2", choices=["NO2", "PM10", "PM25"])
    parser.add_argument("--no-save",   action="store_true")
    args = parser.parse_args()

    from src.data.build_dataset import build_full_dataset
    from src.features.feature_engineering import FeatureEngineer, temporal_split, temporal_split_v2

    model_version = "v2" if args.meteo else "v1"
    split_fn = temporal_split_v2 if args.meteo else temporal_split

    log.info(f"=== Entraînement XGBoost {model_version} | polluant={args.pollutant} ===")

    df_raw = build_full_dataset(
        pollutants=[args.pollutant],
        meteo_path=args.meteo,
    )

    fe = FeatureEngineer(with_meteo=bool(args.meteo))
    df_feat = fe.fit_transform(df_raw[df_raw["pollutant"] == args.pollutant])

    train, val, test = split_fn(df_feat)
    X_train, y_train = fe.get_X_y(train)
    X_val,   y_val   = fe.get_X_y(val)
    X_test,  y_test  = fe.get_X_y(test)

    results, best_model = run_benchmark(
        train, val, test,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        pollutant=args.pollutant,
        model_version=model_version,
        save_model=not args.no_save,
    )

    print("\n" + "="*70)
    print(f"RÉSULTATS — {args.pollutant} — modèle {model_version}")
    print("="*70)
    cols = ["modèle", "set", "MAE", "RMSE", "MAPE", "recall_exceed", "precision_exceed"]
    print(results[[c for c in cols if c in results.columns]].to_string(index=False))
    print("="*70)
