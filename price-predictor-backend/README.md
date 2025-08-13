# Price Predictor – Backend (Flask)

## Env vars
- `BASE_URL` → raw GitHub base that hosts `/models/*.pkl`, `/schemas/*.json`, `/data/*.csv`
  - example: `https://raw.githubusercontent.com/<user>/<repo>/main/price-predictor-backend`
- `GOOGLE_AI_API_KEY` (optional) → Gemini explanations
- `PORT` (Render auto-sets)

## Run (Render)
- Build Command: `pip install -r price-predictor-backend/requirements.txt`
- Start Command: `cd price-predictor-backend && gunicorn app:app -b 0.0.0.0:$PORT`
