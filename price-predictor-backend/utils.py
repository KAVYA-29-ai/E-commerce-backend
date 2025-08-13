import pandas as pd
import joblib
import requests
from io import BytesIO, StringIO
import os

# ------------ CONFIG ------------
BASE_URL = os.getenv(
    "BASE_URL",
    "https://raw.githubusercontent.com/yourusername/price-predictor/main"
)
MODELS_URL = f"{BASE_URL}/models/"

# simple in-process caches
_cached_models = {}
_cached_schemas = {}
_cached_data = {}
_cached_categories = None

# ------------ CATEGORY DISCOVERY ------------
def get_categories_from_github():
    """
    Discover categories by listing .pkl files in models/ via GitHub API.
    Works with public repos. Falls back to default list if API fails/rate-limited.
    """
    global _cached_categories
    if _cached_categories:
        return _cached_categories

    try:
        api_url = MODELS_URL.replace(
            "https://raw.githubusercontent.com/",
            "https://api.github.com/repos/"
        ).replace("/main", "/contents/models")

        resp = requests.get(api_url, timeout=20)
        resp.raise_for_status()
        files = resp.json()

        # rate limit / error body is dict
        if isinstance(files, dict) and str(files.get("message", "")).lower().startswith("api rate limit"):
            _cached_categories = ['phones', 'laptops', 'furniture']
            return _cached_categories

        _cached_categories = [
            f["name"].replace(".pkl", "")
            for f in files
            if isinstance(f, dict) and f.get("name", "").endswith(".pkl")
        ]
        if not _cached_categories:
            _cached_categories = ['phones', 'laptops', 'furniture']
        return _cached_categories
    except Exception:
        _cached_categories = ['phones', 'laptops', 'furniture']
        return _cached_categories

# ------------ LOAD ARTIFACTS ------------
def load_models_and_schemas():
    """Download models, schemas, and CSV market data from GitHub (BASE_URL)."""
    global _cached_models, _cached_schemas, _cached_data
    if _cached_models and _cached_schemas and _cached_data:
        return _cached_models, _cached_schemas, _cached_data

    categories = get_categories_from_github()

    for category in categories:
        # model
        try:
            m = requests.get(f"{BASE_URL}/models/{category}.pkl", timeout=60)
            m.raise_for_status()
            _cached_models[category] = joblib.load(BytesIO(m.content))
            print(f"✅ model: {category}")
        except Exception as e:
            print(f"❌ model load failed ({category}): {e}")

        # schema
        try:
            s = requests.get(f"{BASE_URL}/schemas/{category}.json", timeout=30)
            s.raise_for_status()
            if "json" not in s.headers.get("Content-Type", ""):
                raise ValueError("schema content-type not json")
            _cached_schemas[category] = s.json()
            print(f"✅ schema: {category}")
        except Exception as e:
            print(f"❌ schema load failed ({category}): {e}")

        # data
        try:
            d = requests.get(f"{BASE_URL}/data/{category}.csv", timeout=30)
            d.raise_for_status()
            _cached_data[category] = pd.read_csv(StringIO(d.text))
            print(f"✅ data: {category} ({len(_cached_data[category])})")
        except Exception as e:
            print(f"❌ data load failed ({category}): {e}")

    return _cached_models, _cached_schemas, _cached_data

# ------------ FEATURES ------------
def prepare_features(data: dict, schema: dict):
    """Build feature vector in schema-defined order, handle encoded categoricals."""
    feats = []
    feature_columns = schema.get("feature_columns", [])
    cat_cols = schema.get("categorical_columns", [])
    encoders = schema.get("encoders", {})

    for col in feature_columns:
        if col.endswith("_encoded"):
            original = col[:-8]  # remove "_encoded"
            if original in cat_cols:
                val = str(data.get(original, ""))
                classes = encoders.get(original, [])
                feats.append(classes.index(val) if val in classes else 0)
            else:
                feats.append(0)
        else:
            feats.append(float(data.get(col, get_default_value(col))))
    return feats

def get_default_value(name: str):
    defaults = {
        "rating": 4.0,
        "discount": 10.0,
        "discount_percentage": 10.0,
        "stock": 50,
        "warranty": 12,
        "screen_size": 6.0,
        "storage": 128,
        "ram": 8,
        "processor_score": 2000,
        "dimensions": 100,
        "weight": 20,
    }
    return defaults.get(name, 0)

# ------------ MARKET AVERAGE ------------
def calculate_market_average(category: str, features: dict, market_data: dict):
    """Compute a simple market-average price using similarity filters."""
    if category not in market_data or market_data[category].empty:
        return 500.0

    try:
        df = market_data[category].copy()

        # brand filter
        if "brand" in features and "brand" in df.columns:
            mask = df["brand"].str.lower() == str(features["brand"]).lower()
            if mask.any():
                df = df[mask]

        # category-specific narrowing
        if category == "phones":
            if "storage" in features and "storage" in df.columns:
                df = df[abs(df["storage"] - float(features["storage"])) <= 128]
            if "ram" in features and "ram" in df.columns:
                df = df[abs(df["ram"] - float(features["ram"])) <= 4]

        elif category == "laptops":
            if "ram" in features and "ram" in df.columns:
                df = df[abs(df["ram"] - float(features["ram"])) <= 8]
            if "storage" in features and "storage" in df.columns:
                df = df[abs(df["storage"] - float(features["storage"])) <= 256]

        elif category == "furniture":
            if "material" in features and "material" in df.columns:
                mask = df["material"].str.lower() == str(features["material"]).lower()
                if mask.any():
                    df = df[mask]

        # broaden if too few
        if len(df) < 5:
            df = market_data[category].copy()
            if "rating" in features and "rating" in df.columns:
                df = df[abs(df["rating"] - float(features["rating"])) <= 1.0]

        if df.empty:
            return float(market_data[category]["price"].mean())

        return float(df["price"].mean())

    except Exception:
        return float(market_data[category]["price"].mean())
