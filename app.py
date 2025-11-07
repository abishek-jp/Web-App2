import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib

st.title("Crop Production and Yield Prediction")
st.write("Enter the details below to predict crop production and yield.")

# ===============================
# ✅ Load model, encoder, columns
# ===============================
try:
    model = joblib.load("xgboost_model.pkl")
    OHE = joblib.load("onehot_encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")

except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# ===============================
# ✅ Input options
# ===============================
state_options = [...]     # your full list here (same as before)
district_options = [...]  # your full district list
crop_options = [...]      # your crop list
season_options = ['Kharif', 'Whole Year', 'Rabi', 'Autumn', 'Summer', 'Winter']

# ===============================
# ✅ User Inputs
# ===============================
state = st.selectbox("State", state_options)
district = st.selectbox("District", district_options)
crop = st.selectbox("Crop", crop_options)
season = st.selectbox("Season", season_options)
area = st.number_input("Area (in hectares)", min_value=0.0, format="%f")

# ===============================
# ✅ Prediction Button
# ===============================
if st.button("Predict"):

    # Build DataFrame
    input_data = pd.DataFrame({
        'State': [state],
        'District': [district],
        'Crop': [crop],
        'Season': [season],
        'Area': [area]
    })

    # Separate types
    categorical_cols = ['State', 'District', 'Crop', 'Season']
    numerical_cols = ['Area']

    # ======================================================
    # ✅ Correct One-Hot Encoding WITH column names
    # ======================================================
    try:
        encoded = OHE.transform(input_data[categorical_cols])
        encoded_cols = OHE.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=input_data.index)

    except Exception as e:
        st.error(f"OHE Error: {e}")
        st.stop()

    # ======================================================
    # ✅ Combine + realign EXACTLY as model expects
    # ======================================================
    input_processed = pd.concat([input_data[numerical_cols], encoded_df], axis=1)

    # Reindex to training columns
    input_processed = input_processed.reindex(columns=feature_cols, fill_value=0)

    # ======================================================
    # ✅ Predict
    # ======================================================
    predicted_production = model.predict(input_processed)[0]

    # Avoid negatives due to tree model edge cases
    predicted_production = max(0, predicted_production)

    predicted_yield = predicted_production / area if area > 0 else 0

    # ======================================================
    # ✅ Output
    # ======================================================
    st.subheader("Prediction Results")
    st.write(f"**Predicted Production:** {predicted_production:.2f} tons")
    st.write(f"**Predicted Yield:** {predicted_yield:.2f} tons/ha")
