from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import google.generativeai as genai

from utils import (
    load_models_and_schemas,
    prepare_features,
    calculate_market_average,
)

app = Flask(__name__)
CORS(app)

# Load models/schemas/data from remote (GitHub) at startup
print("🚀 Loading models & schemas from remote …")
models, schemas, market_data = load_models_and_schemas()
print(f"✅ Models loaded: {list(models.keys())}")

# Optional: Gemini for explanations
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
ai_model = None
if GOOGLE_AI_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_AI_API_KEY)
        ai_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Google AI configured")
    except Exception as e:
        print(f"⚠️ Google AI config failed: {e}")

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys()),
        "ai_enabled": bool(ai_model),
    })

@app.get("/categories")
def categories():
    info = {}
    for cat, schema in schemas.items():
        info[cat] = {
            "name": cat.title(),
            "features": schema.get("feature_columns", []),
            "categorical": schema.get("categorical_columns", []),
            "model_info": schema.get("model_info", {}),
        }
    return jsonify({"categories": info})

@app.get("/schema/<category>")
def get_schema(category):
    if category not in schemas:
        return jsonify({"error": "Category not found"}), 404
    return jsonify(schemas[category])

@app.post("/predict/<category>")
def predict(category):
    if category not in models:
        return jsonify({"error": "Category not found"}), 404
    try:
        data = request.get_json(force=True) or {}
        schema = schemas.get(category, {})
        feats = prepare_features(data, schema)
        y_hat = float(models[category].predict([feats])[0])

        market_avg = calculate_market_average(category, data, market_data)
        model_info = schema.get("model_info", {})
        confidence = max(0.6, min(0.95, model_info.get("r2_score", 0.8)))

        resp = {
            "category": category,
            "predicted_price": round(y_hat, 2),
            "market_average": round(float(market_avg), 2),
            "confidence": round(confidence, 2),
            "comparison": {
                "vs_market": "above" if y_hat > market_avg else "below",
                "difference": round(abs(y_hat - market_avg), 2),
                "percentage": round(abs(y_hat - market_avg) / max(market_avg, 1e-6) * 100, 1),
            },
            "features_used": data,
        }
        return jsonify(resp)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/explain")
def explain():
    if not ai_model:
        return jsonify({"error": "AI explanation service not available"}), 503
    try:
        payload = request.get_json(force=True) or {}
        category = payload.get("category")
        predicted_price = payload.get("predicted_price")
        market_average = payload.get("market_average")
        features = payload.get("features", {})
        confidence = payload.get("confidence", 0.8)

        prompt = f"""
You are a friendly AI explaining a price prediction for a {category}.
- Predicted Price: ${predicted_price}
- Market Average: ${market_average}
- Model Confidence: {confidence*100:.0f}%
- Product Features: {json.dumps(features, indent=2)}

Write 2 short paragraphs:
1) How this compares to the market and why
2) Which features likely influenced price + a simple buying tip
Keep it concise, helpful, non-technical.
"""
        out = ai_model.generate_content(prompt)
        return jsonify({"explanation": out.text, "category": category, "confidence": confidence})
    except Exception as e:
        print(f"❌ Explanation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/market-stats/<category>")
def market_stats(category):
    df = market_data.get(category)
    if df is None or df.empty:
        return jsonify({"error": "Category not found or data empty"}), 404
    try:
        stats = {
            "category": category,
            "total_products": int(len(df)),
            "price_stats": {
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
                "mean": float(df["price"].mean()),
                "median": float(df["price"].median()),
                "std": float(df["price"].std()),
            },
            "rating_stats": {
                "min": float(df["rating"].min()) if "rating" in df.columns else None,
                "max": float(df["rating"].max()) if "rating" in df.columns else None,
                "mean": float(df["rating"].mean()) if "rating" in df.columns else None,
            },
            "top_brands": (
                df["brand"].value_counts().head(5).to_dict() if "brand" in df.columns else {}
            ),
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
