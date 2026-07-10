# ============================================================================
# StormSentinel Backend — Inference
# Feature engineering here MUST exactly match v2_05_feature_engineering.py.
# Any drift reintroduces the train/serve skew bug that broke v1's heat
# scores during a real heat wave. Every transformation below is a direct
# port, not a reimplementation.
# ============================================================================

import json
import math
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import torch

from app.ml.model import StormSentinelNetV2, HEAD_ORDER, US_ONLY_HAZARDS, LOW_CONFIDENCE_WILDFIRE_CITIES
from app.ml.geocoding import get_region_for_country

MODEL_DIR = "."  # model artifacts live alongside app/ — see README

# Same expanded variable set as v2 training. If Open-Meteo ever rejects one
# of these, the error will name the exact bad parameter (same defensive
# pattern used throughout the training pipeline) rather than failing silently.
DAILY_VARS = [
    "temperature_2m_max", "apparent_temperature_max",
    "precipitation_sum",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "relative_humidity_2m_mean", "dew_point_2m_mean",
    "surface_pressure_mean", "cloud_cover_mean",
    "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
    "et0_fao_evapotranspiration", "shortwave_radiation_sum",
]

SOUTHERN_HEMISPHERE_CITIES = ["Sydney", "Canberra", "Santiago", "Cape Town", "Perth", "Auckland"]

# Only ~35 days needed now (vs v1's 395-day fetch) — climate_normals_v2.json
# ships the true baseline directly, so there's no live-approximation to
# support with extra history. 35 days comfortably covers the longest
# rolling window (precip_14d_sum) with buffer.
WEATHER_LOOKBACK_DAYS = 35


class ModelArtifacts:
    """Loaded once at app startup, reused across requests."""
    def __init__(self):
        with open(f"{MODEL_DIR}/feature_columns_v2.json") as f:
            self.meta = json.load(f)
        self.scaler = joblib.load(f"{MODEL_DIR}/feature_scaler_v2.pkl")
        with open(f"{MODEL_DIR}/climate_normals_v2.json") as f:
            self.climate_normals = json.load(f)

        self.model = StormSentinelNetV2(len(self.meta["feature_columns"]))
        self.model.load_state_dict(torch.load(f"{MODEL_DIR}/stormsentinel_model_v2.pt", map_location="cpu"))
        self.model.eval()


_artifacts: ModelArtifacts | None = None


def get_artifacts() -> ModelArtifacts:
    global _artifacts
    if _artifacts is None:
        _artifacts = ModelArtifacts()
    return _artifacts


def fetch_weather(lat: float, lon: float, days: int = WEATHER_LOOKBACK_DAYS) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(start_date), "end_date": str(end_date),
        "daily": ",".join(DAILY_VARS), "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        try:
            reason = r.json().get("reason", r.text)
        except Exception:
            reason = r.text
        raise ValueError(f"Open-Meteo rejected the request: {reason}")
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "date"})


def find_nearest_climate_normal(lat: float, lon: float, climate_normals: dict, max_dist_deg: float = 0.5):
    """Same ~55km nearest-trained-city matching used in v1's climate normals fix."""
    best_city, best_dist = None, float("inf")
    for city_name, info in climate_normals.items():
        d = math.dist((info["lat"], info["lon"]), (lat, lon))
        if d < best_dist:
            best_city, best_dist = city_name, d
    if best_city and best_dist < max_dist_deg:
        return climate_normals[best_city]
    return None


def engineer_features(wx_df: pd.DataFrame, lat: float, lon: float, region: str | None, meta: dict, climate_normals: dict):
    """
    Rebuilds the exact feature set from v2_05_feature_engineering.py for the
    most recent day in wx_df. Returns (feature_row_df, raw_last_row) — the
    scaled/ordered features for the model, and the raw values for display.
    """
    df = wx_df.copy().sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month

    # Hemisphere-adjusted seasonality — city name isn't known for arbitrary
    # searches, so use a latitude-based rule instead (equivalent for this
    # purpose: south of the equator = Southern Hemisphere).
    is_southern = lat < 0
    if is_southern:
        df["hemisphere_month"] = ((df["month"] + 5) % 12) + 1
    else:
        df["hemisphere_month"] = df["month"]
    df["month_sin"] = np.sin(2 * np.pi * df["hemisphere_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["hemisphere_month"] / 12)

    # Dewpoint depression — real ERA5 dewpoint, matches training exactly
    df["dewpoint_depression"] = df["temperature_2m_max"] - df["dew_point_2m_mean"]

    # Wind features
    wind_rad = np.radians(df["winddirection_10m_dominant"])
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)
    df["wind_gust_ratio"] = df["windgusts_10m_max"] / (df["windspeed_10m_max"] + 1.0)
    df["wind_humidity_interaction"] = df["windspeed_10m_max"] * (100 - df["relative_humidity_2m_mean"]) / 100

    # Rolling / lag features — identical windows to training
    df["humidity_3d_avg"]  = df["relative_humidity_2m_mean"].rolling(3, min_periods=1).mean()
    df["humidity_7d_avg"]  = df["relative_humidity_2m_mean"].rolling(7, min_periods=1).mean()
    df["precip_7d_sum"]    = df["precipitation_sum"].rolling(7, min_periods=1).sum()
    df["precip_14d_sum"]   = df["precipitation_sum"].rolling(14, min_periods=1).sum()
    df["et0_7d_sum"]       = df["et0_fao_evapotranspiration"].rolling(7, min_periods=1).sum()
    df["temp_change_1d"]   = df["temperature_2m_max"].diff().fillna(0)

    is_dry = df["precipitation_sum"] < 1
    df["days_since_rain"] = is_dry.groupby((~is_dry).cumsum()).cumsum()

    is_hot = df["apparent_temperature_max"] >= 35
    df["heat_streak"] = is_hot.groupby((~is_hot).cumsum()).cumsum()

    df["pressure_change_1d"] = df["surface_pressure_mean"].diff().fillna(0)
    df["pressure_change_3d"] = df["surface_pressure_mean"].diff(3).fillna(0)

    df["cloud_cover_3d_avg"] = df["cloud_cover_mean"].rolling(3, min_periods=1).mean()
    is_clear = df["cloud_cover_mean"] < 30
    df["consecutive_clear_days"] = is_clear.groupby((~is_clear).cumsum()).cumsum()

    df["soil_moisture_7d_trend"] = df["soil_moisture_0_to_7cm_mean"].diff(7).fillna(0)

    # Temperature anomaly — TRUE baseline lookup, not a live approximation.
    # This is the fix for v1's flatlined-heat-score bug, built in from day one.
    normal = find_nearest_climate_normal(lat, lon, climate_normals)
    current_month = str(int(df["month"].iloc[-1]))
    if normal and current_month in normal["monthly_avg_max_temp"]:
        baseline = normal["monthly_avg_max_temp"][current_month]
    else:
        baseline = None  # genuinely novel location — no live approximation fallback (that's what caused the bug)

    if baseline is not None:
        df["temp_anomaly"] = df["temperature_2m_max"] - baseline
    else:
        df["temp_anomaly"] = 0.0  # honest neutral value, not a guess

    # Region one-hot — zero everywhere if outside the training footprint
    region_cols = meta["region_dummy_columns"]
    for col in region_cols:
        df[col] = 0
    if region:
        region_col = f"region_{region}"
        if region_col in region_cols:
            df[region_col] = 1
        # else: region is a valid trained category but happens to be the
        # dropped reference level (Africa) — correctly stays all-zero.

    # Align to the exact training column order — never reconstruct order by
    # hand, always follow the canonical list from feature_columns_v2.json.
    feature_columns = meta["feature_columns"]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0  # only true for a handful of static/edge-case columns

    last_row = df.iloc[[-1]][feature_columns].fillna(0)
    return last_row, df.iloc[-1], baseline is not None


def run_inference(feature_row: pd.DataFrame, artifacts: ModelArtifacts) -> dict:
    numeric_cols = artifacts.meta["numeric_columns"]
    feat = feature_row.copy()
    feat[numeric_cols] = artifacts.scaler.transform(feat[numeric_cols])
    x = torch.tensor(feat.values.astype(np.float32))
    with torch.no_grad():
        outputs = artifacts.model(x)
    return {name: float(torch.sigmoid(outputs[name]).item()) for name in HEAD_ORDER}


def predict_for_location(city: str, country_code: str, lat: float, lon: float, region: str | None) -> dict:
    """
    Full pipeline: fetch weather -> engineer features -> run model ->
    apply documented low-confidence flags. This is the single entry point
    the /predict endpoint calls.
    """
    artifacts = get_artifacts()
    wx_df = fetch_weather(lat, lon)
    feature_row, raw_last_row, has_true_baseline = engineer_features(
        wx_df, lat, lon, region, artifacts.meta, artifacts.climate_normals
    )
    probs = run_inference(feature_row, artifacts)
    scores = {name: round(p * 100) for name, p in probs.items()}

    is_us = country_code == "US"

    warnings = []
    if not is_us:
        warnings.append({
            "type": "region_unvalidated",
            "hazards": sorted(US_ONLY_HAZARDS),
            "message": (
                "Tornado, Hail, and Thunderstorm Wind were trained exclusively on US "
                "storm data and have no real grounding outside the US. Wildfire and "
                "Heat use physically-grounded features that transfer globally."
            ),
        })
    if city in LOW_CONFIDENCE_WILDFIRE_CITIES:
        warnings.append({
            "type": "low_confidence_wildfire",
            "hazards": ["wildfire"],
            "message": (
                f"{city}'s wildfire label showed persistent data-quality issues during "
                f"training (likely industrial/flare contamination in satellite fire "
                f"detection) that couldn't be cleanly resolved. Treat this score as "
                f"lower-confidence."
            ),
        })
    if not has_true_baseline:
        warnings.append({
            "type": "no_climate_baseline",
            "hazards": ["heat"],
            "message": (
                "This location is far from any of the 55 cities used in training, so "
                "heat risk uses a neutral seasonal baseline rather than this specific "
                "location's true climate normal."
            ),
        })

    return {
        "scores": scores,
        "composite_score": round(sum(scores.values()) / len(scores)),
        "raw_weather": {
            "temperature_max": round(float(raw_last_row["temperature_2m_max"]), 1),
            "apparent_temperature_max": round(float(raw_last_row["apparent_temperature_max"]), 1),
            "humidity": round(float(raw_last_row["relative_humidity_2m_mean"])),
            "wind_speed": round(float(raw_last_row["windspeed_10m_max"])),
            "wind_gusts": round(float(raw_last_row["windgusts_10m_max"])),
            "precipitation": round(float(raw_last_row["precipitation_sum"]), 1),
            "cloud_cover": round(float(raw_last_row["cloud_cover_mean"])),
        },
        "warnings": warnings,
        "is_us": is_us,
    }
