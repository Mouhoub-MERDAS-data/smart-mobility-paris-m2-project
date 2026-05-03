"""
Pipeline de construction du dataset complet.

Fusionne :
  - Données pollution Airparif (NO2, PM10, PM2.5) par segment du périphérique
  - Données météo journalières interpolées en horaire
  - Jours fériés

Usage :
    python -m src.data.build_dataset

    ou depuis Python :
        from src.data.build_dataset import build_full_dataset
        df = build_full_dataset()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins par défaut (relatifs à la racine du projet)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

POLLUTION_FILES = {
    "NO2":  [
        DATA_DIR / "2024_NO2_boulevard_périphérique.csv",
        DATA_DIR / "2025_NO2_boulevard_périphérique.csv",
        DATA_DIR / "2026_NO2_boulevard_périphérique.csv",
    ],
    "PM10": [
        DATA_DIR / "2024_PM10_boulevard_périphérique.csv",
        DATA_DIR / "2025_PM10_boulevard_périphérique.csv",
        DATA_DIR / "2026_PM10_boulevard_périphérique.csv",
    ],
    "PM25": [
        DATA_DIR / "2024_PM25_boulevard_périphérique.csv",
        DATA_DIR / "2025_PM25_boulevard_périphérique.csv",
        DATA_DIR / "2026_PM25_boulevard_périphérique.csv",
    ],
}

HOLIDAYS_FILE = DATA_DIR / "jours_feries_metropole (1).csv"

SEGMENTS = [
    "Chap-Bagn", "Bagn-Berc", "Berc-Ital", "Ital-A6a",
    "A6a-Sevr",  "Sevr-Aute", "Aute-Mail", "Mail-Chap",
]

# Colonnes météo à interpoler (daily → hourly)
METEO_COLS_NUMERIC = [
    "MAX_TEMPERATURE_C", "MIN_TEMPERATURE_C",
    "TEMPERATURE_MORNING_C_6H", "TEMPERATURE_NOON_C_12H",
    "TEMPERATURE_EVENING_C_18H", "TEMPERATURE_NIGHT_C_3H",
    "PRECIP_TOTAL_DAY_MM", "HUMIDITY_MAX_PERCENT",
    "PRESSURE_MAX_MB", "CLOUDCOVER_AVG_PERCENT",
    "WINDSPEED_MAX_KMH", "VISIBILITY_AVG_KM",
    "UV_INDEX", "SUNHOUR", "TOTAL_SNOW_MM",
    "HEATINDEX_MAX_C", "DEWPOINT_MAX_C",
]


# ---------------------------------------------------------------------------
# 1. Pollution
# ---------------------------------------------------------------------------

def load_pollution(pollutants: list[str] | None = None) -> pd.DataFrame:
    """Charge et concatène tous les CSV pollution, retourne format long."""
    if pollutants is None:
        pollutants = list(POLLUTION_FILES.keys())

    frames = []
    for pollutant in pollutants:
        for fpath in POLLUTION_FILES[pollutant]:
            # Gérer le fait que le fichier peut exister ou non selon l'OS/encodage
            candidates = list(DATA_DIR.glob(f"*{fpath.stem.split('_')[0]}*{pollutant}*"))
            # Cherche aussi avec le nom exact
            if fpath.exists():
                candidates = [fpath]
            elif not candidates:
                log.warning(f"Fichier introuvable : {fpath.name} — ignoré")
                continue

            f = candidates[0]
            df = pd.read_csv(f, parse_dates=["time"])
            df["pollutant"] = pollutant
            frames.append(df)
            log.info(f"Chargé {f.name} : {len(df)} lignes")

    if not frames:
        raise FileNotFoundError(
            "Aucun fichier pollution trouvé. Vérifiez DATA_DIR et les noms de fichiers."
        )

    raw = pd.concat(frames, ignore_index=True)

    # Format long : une ligne = (time, segment, pollutant, value)
    df_long = raw.melt(
        id_vars=["time", "pollutant"],
        value_vars=[c for c in raw.columns if c in SEGMENTS],
        var_name="segment",
        value_name="value",
    )
    df_long = df_long.sort_values(["pollutant", "segment", "time"]).reset_index(drop=True)
    log.info(f"Pollution totale : {len(df_long):,} lignes | "
             f"{df_long['time'].min().date()} → {df_long['time'].max().date()}")
    return df_long


# ---------------------------------------------------------------------------
# 2. Météo
# ---------------------------------------------------------------------------

def load_and_interpolate_meteo(meteo_path: str | Path) -> pd.DataFrame:
    """
    Charge la météo journalière et l'interpole en horaire.

    Le fichier attendu a une colonne DATE (YYYY-MM-DD ou DD/MM/YYYY)
    et des colonnes numériques journalières.

    Retourne un DataFrame avec colonne 'time' (datetime horaire) et
    toutes les colonnes météo interpolées.
    """
    meteo_path = Path(meteo_path)
    if not meteo_path.exists():
        raise FileNotFoundError(f"Fichier météo introuvable : {meteo_path}")

    # Détection automatique du format : Excel (.xlsx / .xls) ou CSV
    suffix = meteo_path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        df = pd.read_excel(meteo_path, sheet_name=0)
        log.info(f"Météo Excel chargée ({suffix}) : {len(df)} jours | colonnes : {df.columns.tolist()}")
    else:
        with open(meteo_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")
        df = pd.read_csv(meteo_path, sep=sep, encoding="utf-8-sig")
        log.info(f"Météo CSV chargée : {len(df)} jours | colonnes : {df.columns.tolist()}")

    # Normaliser le nom de la colonne date
    date_col = next((c for c in df.columns if "DATE" in c.upper() or c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"Colonne DATE introuvable dans {meteo_path.name}. "
                         f"Colonnes disponibles : {df.columns.tolist()}")
    df = df.rename(columns={date_col: "date"})

    # Parser la date (formats courants : YYYY-MM-DD ou DD/MM/YYYY)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    else:
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

    df = df.set_index("date").sort_index()

    # Colonnes numériques disponibles
    available_cols = [c for c in METEO_COLS_NUMERIC if c in df.columns]
    missing = [c for c in METEO_COLS_NUMERIC if c not in df.columns]
    if missing:
        log.warning(f"Colonnes météo manquantes (ignorées) : {missing}")

    # Interpolation daily → hourly via reindex + interpolation linéaire
    # Créer un index horaire couvrant toute la période
    hourly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max() + pd.Timedelta(hours=23),
        freq="h",
    )

    df_hourly = (
        df[available_cols]
        .reindex(df.index.union(hourly_index))
        .interpolate(method="time")
        .reindex(hourly_index)
    )

    # Features météo dérivées
    if "MAX_TEMPERATURE_C" in df_hourly.columns and "MIN_TEMPERATURE_C" in df_hourly.columns:
        df_hourly["TEMP_RANGE_C"] = df_hourly["MAX_TEMPERATURE_C"] - df_hourly["MIN_TEMPERATURE_C"]

    if "MAX_TEMPERATURE_C" in df_hourly.columns:
        df_hourly["IS_COLD"] = (df_hourly["MAX_TEMPERATURE_C"] < 5).astype(int)
        df_hourly["IS_HOT"]  = (df_hourly["MAX_TEMPERATURE_C"] > 25).astype(int)

    if "PRECIP_TOTAL_DAY_MM" in df_hourly.columns:
        df_hourly["IS_RAINY"] = (df_hourly["PRECIP_TOTAL_DAY_MM"] > 1.0).astype(int)

    if "TOTAL_SNOW_MM" in df_hourly.columns:
        df_hourly["IS_SNOWY"] = (df_hourly["TOTAL_SNOW_MM"] > 0.0).astype(int)

    if "CLOUDCOVER_AVG_PERCENT" in df_hourly.columns:
        df_hourly["IS_OVERCAST"] = (df_hourly["CLOUDCOVER_AVG_PERCENT"] > 80).astype(int)

    df_hourly = df_hourly.reset_index().rename(columns={"index": "time"})
    log.info(f"Météo interpolée : {len(df_hourly):,} heures | "
             f"{df_hourly['time'].min().date()} → {df_hourly['time'].max().date()}")
    return df_hourly


# ---------------------------------------------------------------------------
# 3. Jours fériés
# ---------------------------------------------------------------------------

def load_holidays(holidays_path: str | Path | None = None) -> set[object]:
    """Retourne un set des dates (datetime.date) de jours fériés."""
    if holidays_path is None:
        holidays_path = HOLIDAYS_FILE
    holidays_path = Path(holidays_path)
    if not holidays_path.exists():
        log.warning("Fichier jours fériés introuvable — colonne is_holiday = 0 partout")
        return set()
    df = pd.read_csv(holidays_path, parse_dates=["date"])
    return set(df["date"].dt.date)


# ---------------------------------------------------------------------------
# 4. Pipeline complet
# ---------------------------------------------------------------------------

def build_full_dataset(
    pollutants: list[str] | None = None,
    meteo_path: str | Path | None = None,
    holidays_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Construit le dataset complet :
        pollution long format + features temporelles + météo + jours fériés.

    Args:
        pollutants    : liste de polluants ['NO2','PM10','PM25']. Défaut = tous.
        meteo_path    : chemin vers le CSV météo journalier.
        holidays_path : chemin vers le CSV jours fériés.
        output_path   : si fourni, sauvegarde le dataset en CSV.

    Returns:
        pd.DataFrame prêt pour le feature engineering.
    """
    if pollutants is None:
        pollutants = ["NO2", "PM10", "PM25"]

    # --- Pollution ---
    df = load_pollution(pollutants)

    # --- Features temporelles de base ---
    df["hour"]        = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day_name"]    = df["time"].dt.day_name()
    df["month"]       = df["time"].dt.month
    df["year"]        = df["time"].dt.year
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # --- Jours fériés ---
    holiday_dates = load_holidays(holidays_path)
    df["is_holiday"] = df["time"].dt.date.isin(holiday_dates).astype(int)

    # --- Météo (merge si disponible) ---
    if meteo_path is not None:
        meteo_hourly = load_and_interpolate_meteo(meteo_path)
        df = df.merge(meteo_hourly, on="time", how="left")
        meteo_cols = [c for c in meteo_hourly.columns if c != "time"]
        n_nan = df[meteo_cols].isnull().sum().sum()
        if n_nan > 0:
            log.warning(
                f"{n_nan} NaN dans les colonnes météo après merge "
                f"(période pollution hors couverture météo — normal pour 2024 si météo=2025)."
            )
        log.info(f"Météo fusionnée : {len(meteo_cols)} features ajoutées")

    # --- Sauvegarde ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(f"Dataset sauvegardé → {output_path} ({len(df):,} lignes)")

    log.info(f"Dataset final : {len(df):,} lignes × {df.shape[1]} colonnes")
    return df


# ---------------------------------------------------------------------------
# Entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construit le dataset complet")
    parser.add_argument("--meteo",     default=None,  help="Chemin fichier météo daily CSV")
    parser.add_argument("--output",    default="data/processed/dataset_complet.csv")
    parser.add_argument("--pollutants", nargs="+", default=["NO2", "PM10", "PM25"])
    args = parser.parse_args()

    df = build_full_dataset(
        pollutants=args.pollutants,
        meteo_path=args.meteo,
        output_path=args.output,
    )
    print(f"\n✅ Dataset prêt : {df.shape}")
    print(df.dtypes)
    print(df.head(3))
