import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Crop Production & Yield Predictor", layout="centered")

# ------------------------------
# 1. LOAD MODEL + ENCODER + COLUMNS
# ------------------------------
model = joblib.load("xgboost_model.pkl")
OHE = joblib.load("onehot_encoder.pkl")
feature_cols = joblib.load("feature_columns.pkl")   # list of final training columns

# ------------------------------
# 2. INPUT FORM
# ------------------------------
st.title("🌾 Crop Production & Yield Predictor")

State = st.selectbox("State", OHE.categories_[0])          # based on training categories
District = st.selectbox("District", OHE.categories_[1])
Crop = st.selectbox("Crop", OHE.categories_[2])
Season = st.selectbox("Season", OHE.categories_[3])

Area = st.number_input("Area (in hectares)", min_value=0.1, format="%.2f")

if st.button("Predict"):
    # ------------------------------
    # 3. BUILD RAW INPUT DF
    # ------------------------------
    raw_input = pd.DataFrame({
        "State": [State],
        "District": [District],
        "Crop": [Crop],
        "Season": [Season],
        "Area": [Area]
    })

    # ------------------------------
    # 4. IDENTIFY CATEGORICAL + NUMERIC
    # ------------------------------
    categorical_cols = ["State", "District", "Crop", "Season"]
    numerical_cols = ["Area"]

    # ------------------------------
    # 5. OHE TRANSFORM USING TRAINED ENCODER
    # ------------------------------
    encoded = OHE.transform(raw_input[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=OHE.get_feature_names_out(categorical_cols),
        index=raw_input.index
    )

    # ------------------------------
    # 6. COMBINE NUMERIC + ENCODED
    # ------------------------------
    input_processed = pd.concat([raw_input[numerical_cols], encoded_df], axis=1)

    # ------------------------------
    # 7. REINDEX TO MATCH TRAINING ORDER
    # ------------------------------
    input_processed = input_processed.reindex(columns=feature_cols, fill_value=0)

    # ------------------------------
    # 8. PREDICTION
    # ------------------------------
    pred_production = model.predict(input_processed)[0]

    # ------------------------------
    # 9. DERIVE yield
    # ------------------------------
    pred_yield = pred_production / Area

    # ------------------------------
    # 10. DISPLAY RESULTS
    # ------------------------------
    st.success(f"✅ **Predicted Production:** {pred_production:.2f} tons")
    st.success(f"✅ **Predicted Yield:** {pred_yield:.2f} tons/ha")
