"""
Feature engineering pour la prédiction de pollution sur le périphérique parisien.

Construit le dataset au format long (une ligne = un segment × un instant) avec :
- features temporelles cycliques (sin/cos heure, jour, mois)
- jours fériés
- features de retard (lag) et fenêtres glissantes (rolling)
- météo (optionnel, période réduite)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = [
    "Chap-Bagn",
    "Bagn-Berc",
    "Berc-Ital",
    "Ital-A6a",
    "A6a-Sevr",
    "Sevr-Aute",
    "Aute-Mail",
    "Mail-Chap",
]

# Seuils légaux annuels OMS / UE pour information dans les visualisations
SEUILS_LEGAUX = {"NO2": 40.0, "PM10": 40.0, "PM25": 25.0}


def load_air_quality(path: str | Path) -> pd.DataFrame:
    """Charge le CSV pollution préparé et passe en format long.

    Le CSV d'entrée est en format wide (une colonne par segment).
    On convertit en format long : (time, segment, pollutant, value).
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df_long = df.melt(
        id_vars=[
            "time",
            "pollutant",
            "hour",
            "day_of_week",
            "day_name",
            "month",
            "is_weekend",
            "is_peak_hour",
        ],
        value_vars=SEGMENTS,
        var_name="segment",
        value_name="value",
    )
    return df_long.sort_values(["pollutant", "segment", "time"]).reset_index(drop=True)


def load_holidays(path: str | Path) -> pd.DataFrame:
    """Charge les jours fériés métropole et retourne un DataFrame indexé par date."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df[["date", "nom_jour_ferie"]]


def load_weather(path: str | Path) -> pd.DataFrame:
    """Charge le CSV Open-Meteo (header sur 3 lignes à sauter)."""
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(
        columns={
            "time": "time",
            "temperature_2m (°C)": "temperature",
            "relative_humidity_2m (%)": "humidity",
            "precipitation (mm)": "precipitation",
            "weather_code (wmo code)": "weather_code",
            "wind_speed_10m (km/h)": "wind_speed",
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["temperature"]).reset_index(drop=True)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour, day_of_week, month en sin/cos pour capter la cyclicité."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_holiday_flag(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne booléenne is_holiday."""
    df = df.copy()
    holiday_dates = set(holidays["date"])
    df["is_holiday"] = df["time"].dt.date.isin(holiday_dates)
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] = (1, 2, 3, 24, 168),
    group_cols: list[str] = ("pollutant", "segment"),
) -> pd.DataFrame:
    """Ajoute des features de retard. lag=1 = h-1, lag=24 = même heure veille,
    lag=168 = même heure semaine précédente.
    """
    df = df.copy().sort_values([*group_cols, "time"])
    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby(list(group_cols))["value"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = (3, 24),
    group_cols: list[str] = ("pollutant", "segment"),
) -> pd.DataFrame:
    """Moyennes glissantes calculées sur les valeurs PASSÉES uniquement (shift(1))
    pour éviter toute fuite (data leakage).
    """
    df = df.copy().sort_values([*group_cols, "time"])
    grouped = df.groupby(list(group_cols))["value"]
    for w in windows:
        df[f"rolling_mean_{w}h"] = (
            grouped.shift(1).rolling(w, min_periods=1).mean().reset_index(drop=True)
        )
    return df


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge la météo sur l'horodatage exact (left join : NaN si pas de météo dispo)."""
    return df.merge(weather, on="time", how="left")


def build_dataset(
    air_quality_path: str | Path,
    holidays_path: str | Path,
    weather_path: str | Path | None = None,
    pollutants: list[str] | None = None,
) -> pd.DataFrame:
    """Pipeline complet : chargement + feature engineering.

    Args:
        air_quality_path: CSV pollution préparé
        holidays_path: CSV jours fériés
        weather_path: CSV Open-Meteo (optionnel, période courte)
        pollutants: liste des polluants à garder (par défaut tous)

    Returns:
        DataFrame en format long, prêt pour la modélisation, avec colonne 'value'
        comme cible.
    """
    df = load_air_quality(air_quality_path)
    if pollutants is not None:
        df = df[df["pollutant"].isin(pollutants)].reset_index(drop=True)

    df = add_cyclical_features(df)
    df = add_holiday_flag(df, load_holidays(holidays_path))
    df = add_lag_features(df)
    df = add_rolling_features(df)

    if weather_path is not None:
        df = add_weather_features(df, load_weather(weather_path))

    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_peak_hour"] = df["is_peak_hour"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)

    return df


def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2024-12-31",
    val_end: str = "2025-06-30",
    time_col: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel rigoureux. Pas de shuffle, pas de fuite future."""
    train = df[df[time_col] <= train_end].copy()
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test = df[df[time_col] > val_end].copy()
    return train, val, test
