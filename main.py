"""
Point d'entrée principal du projet Smart Mobility Paris.

Commandes disponibles :

    # 1. Construire le dataset et entraîner le modèle (sans météo)
    python main.py

    # 2. Entraîner avec la météo 2025
    python main.py --meteo data/meteo_2025.csv

    # 3. Entraîner pour un polluant spécifique
    python main.py --meteo data/meteo_2025.csv --pollutant PM10

    # 4. Entraîner tous les polluants
    python main.py --meteo data/meteo_2025.csv --all-pollutants

    # 5. Lancer le dashboard Streamlit
    python main.py --dashboard

    # 6. Tout faire (entraîner + lancer le dashboard)
    python main.py --meteo data/meteo_2025.csv --dashboard
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_pipeline(
    pollutant: str,
    meteo_path: str | None,
    save: bool = True,
) -> None:
    """Lance le pipeline complet pour un polluant."""
    from src.data.build_dataset import build_full_dataset
    from src.features.feature_engineering import FeatureEngineer, temporal_split, temporal_split_v2
    from src.models.train import run_benchmark

    model_version = "v2" if meteo_path else "v1"
    split_fn = temporal_split_v2 if meteo_path else temporal_split

    log.info("=" * 60)
    log.info(f"PIPELINE — polluant={pollutant} | version={model_version}")
    log.info("=" * 60)

    # 1. Charger et fusionner les données
    log.info("Étape 1/4 — Construction du dataset...")
    df_raw = build_full_dataset(
        pollutants=[pollutant],
        meteo_path=meteo_path,
        output_path=PROJECT_ROOT / f"data/processed/dataset_{pollutant}_{model_version}.csv",
    )

    # 2. Feature engineering
    log.info("Étape 2/4 — Feature engineering...")
    df_poll = df_raw[df_raw["pollutant"] == pollutant].copy()
    fe = FeatureEngineer(with_meteo=bool(meteo_path))
    df_feat = fe.fit_transform(df_poll)

    log.info(f"Features effectives ({len(fe.feature_names)}) : {fe.feature_names[:8]}...")
    if meteo_path:
        meteo_feats = [f for f in fe.feature_names if any(
            k in f for k in ["TEMP", "HUMID", "PRECIP", "PRESSURE", "CLOUD", "WIND"]
        )]
        log.info(f"Features météo incluses ({len(meteo_feats)}) : {meteo_feats}")

    # 3. Split temporel
    log.info("Étape 3/4 — Split temporel...")
    train, val, test = split_fn(df_feat)
    X_train, y_train = fe.get_X_y(train)
    X_val,   y_val   = fe.get_X_y(val)
    X_test,  y_test  = fe.get_X_y(test)

    log.info(
        f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    # 4. Entraînement + benchmark
    log.info("Étape 4/4 — Entraînement et benchmark...")
    results, best_model = run_benchmark(
        train, val, test,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        pollutant=pollutant,
        model_version=model_version,
        save_model=save,
    )

    # Résultats finaux
    print("\n" + "=" * 70)
    print(f"RÉSULTATS FINAUX — {pollutant} — modèle {model_version}")
    print("=" * 70)
    cols = ["modèle", "set", "MAE", "RMSE", "MAPE", "recall_exceed", "f1_exceed"]
    cols_avail = [c for c in cols if c in results.columns]
    print(results[cols_avail].to_string(index=False))
    print("=" * 70)

    # Résumé
    test_xgb = results[
        (results["modèle"].str.contains("XGBoost")) & (results["set"] == "test")
    ]
    test_pers = results[
        (results["modèle"] == "Persistance (h-1)") & (results["set"] == "test")
    ]
    if not test_xgb.empty and not test_pers.empty:
        gain = (test_pers["MAE"].values[0] - test_xgb["MAE"].values[0]) / test_pers["MAE"].values[0] * 100
        log.info(f"✅ Gain XGBoost vs persistance : {gain:.1f}% de réduction MAE")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Mobility Paris — Pipeline ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--meteo", default=None,
        help="Chemin vers le fichier météo journalier (CSV). Active le modèle V2."
    )
    parser.add_argument(
        "--pollutant", default="NO2", choices=["NO2", "PM10", "PM25"],
        help="Polluant à modéliser (défaut : NO2)"
    )
    parser.add_argument(
        "--all-pollutants", action="store_true",
        help="Entraîne un modèle pour chaque polluant (NO2, PM10, PM25)"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Lance le dashboard Streamlit après l'entraînement"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Ne sauvegarde pas les modèles (debug)"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Lance uniquement le dashboard (sans entraîner)"
    )
    args = parser.parse_args()

    if args.meteo and not Path(args.meteo).exists():
        log.error(f"Fichier météo introuvable : {args.meteo}")
        sys.exit(1)

    if not args.dashboard_only:
        pollutants = ["NO2", "PM10", "PM25"] if args.all_pollutants else [args.pollutant]
        for p in pollutants:
            train_pipeline(
                pollutant=p,
                meteo_path=args.meteo,
                save=not args.no_save,
            )

    if args.dashboard or args.dashboard_only:
        streamlit_path = PROJECT_ROOT / "interface" / "streamlit_app.py"
        if not streamlit_path.exists():
            log.error(f"streamlit_app.py introuvable : {streamlit_path}")
            sys.exit(1)
        log.info("Lancement du dashboard Streamlit...")
        log.info("➡️  Ouvre http://localhost:8501 dans ton navigateur")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(streamlit_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
