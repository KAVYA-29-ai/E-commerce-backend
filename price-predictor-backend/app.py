from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

# optional (only used for /explain if key present)
try:
    import google.generativeai as genai
except Exception:
    genai = None

from utils import (
    load_models_and_schemas,
    prepare_features,
    calculate_market_average,
    get_categories_from_github,
)

app = Flask(__name__)
CORS(app)

# ---------- Config ----------
BASE_URL = os.getenv(
    "BASE_URL",
    "https://raw.githubusercontent.com/yourusername/price-predictor/main"
)

# Google Generative AI (optional)
ai_model = None
API_KEY = os.getenv("GOOGLE_AI_API_KEY")
if genai and API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        ai_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Google AI configured")
    except Exception as e:
        print(f"⚠️ Google AI init failed: {e}")
else:
    print("ℹ️ AI explanations disabled (no GOOGLE_AI_API_KEY or library).")

# ---------- Load artifacts from GitHub ----------
print("🚀 Loading models & schemas from:", BASE_URL)
models, schemas, market_data = load_models_and_schemas()
print("✅ Models loaded:", list(models.keys()))

# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "base_url": BASE_URL,
        "categories": list(models.keys()),
        "ai_enabled": ai_model is not None
    })

@app.route("/categories", methods=["GET"])
def categories():
    # dynamic discovery (reads models/*.pkl in the repo)
    discovered = get_categories_from_github()
    return jsonify({"categories": discovered})

@app.route("/schema/<category>", methods=["GET"])
def schema(category):
    if category not in schemas:
        return jsonify({"error": "Category not found"}), 404
    return jsonify(schemas[category])

@app.route("/predict/<category>", methods=["POST"])
def predict(category):
    if category not in models or category not in schemas:
        return jsonify({"error": "Category not found"}), 404

    try:
        payload = request.get_json(force=True) or {}
        model = models[category]
        schema = schemas[category]

        # features for model prediction
        feature_vec = prepare_features(payload, schema)
        pred = float(model.predict([feature_vec])[0])

        # market average using CSV
        avg = calculate_market_average(category, payload, market_data)

        r2 = schemas[category].get("model_info", {}).get("r2_score", 0.8)
        confidence = max(0.6, min(0.95, float(r2)))

        out = {
            "category": category,
            "predicted_price": round(pred, 2),
            "market_average": round(avg, 2),
            "confidence": round(confidence, 2),
            "comparison": {
                "vs_market": "above" if pred > avg else "below",
                "difference": round(abs(pred - avg), 2),
                "percentage": round(abs(pred - avg) / (avg or 1) * 100, 2)
            }
        }
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/market-stats/<category>", methods=["GET"])
def market_stats(category):
    if category not in market_data:
        return jsonify({"error": "Category not found"}), 404
    try:
        df = market_data[category]
        stats = {
            "category": category,
            "total_products": int(len(df)),
            "price": {
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
                "mean": float(df["price"].mean()),
                "median": float(df["price"].median()),
                "std": float(df["price"].std())
            },
            "rating": {
                "min": float(df["rating"].min()),
                "max": float(df["rating"].max()),
                "mean": float(df["rating"].mean())
            }
        }
        if "brand" in df.columns:
            stats["top_brands"] = df["brand"].value_counts().head(5).to_dict()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/explain", methods=["POST"])
def explain():
    if not ai_model:
        return jsonify({"error": "AI explanation disabled"}), 503

    try:
        data = request.get_json(force=True) or {}
        category = data.get("category")
        predicted_price = data.get("predicted_price")
        market_average = data.get("market_average")
        features = data.get("features", {})
        confidence = data.get("confidence", 0.8)

        prompt = f"""
You are an assistant explaining a price prediction for a {category}.
Predicted Price: ${predicted_price}
Market Average: ${market_average}
Model Confidence: {confidence*100:.0f}%
Features: {json.dumps(features, indent=2)}

Explain in 2 short paragraphs:
- how it compares to market average
- key features that likely influenced price
- a quick buy recommendation
Keep it friendly and concise.
"""
        resp = ai_model.generate_content(prompt)
        return jsonify({"explanation": resp.text, "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
