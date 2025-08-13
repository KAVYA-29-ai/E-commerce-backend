import streamlit as st
import pandas as pd
from utils import load_models_and_schemas

# ========================================
# LOAD MODELS / SCHEMAS / DATA FROM GITHUB
# ========================================
st.set_page_config(page_title="Multi-Product Price Predictor", layout="centered")
st.title("📊 Multi-Product Price Predictor")
st.write("Predict prices for multiple product categories dynamically loaded from GitHub.")

with st.spinner("Loading models and data from GitHub..."):
    models, schemas, market_data = load_models_and_schemas()

# ========================================
# CATEGORY SELECTION
# ========================================
category = st.selectbox("Select a product category", list(models.keys()))

if not category:
    st.warning("No categories found. Please check your GitHub repository setup.")
    st.stop()

st.subheader(f"Category: {category.capitalize()}")

# ========================================
# FEATURE INPUT FORM
# ========================================
input_schema = schemas.get(category, {})
if not input_schema:
    st.error(f"No schema found for category '{category}'. Cannot proceed.")
    st.stop()

user_input = {}
with st.form("prediction_form"):
    st.write("Enter product details:")
    for feature, dtype in input_schema.items():
        if dtype == "int":
            user_input[feature] = st.number_input(feature, step=1, format="%d")
        elif dtype == "float":
            user_input[feature] = st.number_input(feature, format="%.2f")
        else:
            user_input[feature] = st.text_input(feature)

    submitted = st.form_submit_button("Predict Price")

# ========================================
# PREDICTION
# ========================================
if submitted:
    try:
        model = models[category]
        df = pd.DataFrame([user_input])
        predicted_price = model.predict(df)[0]

        # Compare with market average
        market_df = market_data.get(category)
        avg_price = market_df['price'].mean() if market_df is not None else None

        st.success(f"Predicted Price: ₹{predicted_price:,.2f}")

        if avg_price is not None:
            if predicted_price > avg_price:
                st.info(f"💡 This price is above the market average of ₹{avg_price:,.2f}")
            elif predicted_price < avg_price:
                st.info(f"💡 This price is below the market average of ₹{avg_price:,.2f}")
            else:
                st.info(f"💡 This price matches the market average.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
