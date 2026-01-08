import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicție preț locuință, București")

# Load artifacts
model = joblib.load("models/model_linreg.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
st.subheader("Introdu datele locuinței")

# INPUTS
rooms_count = st.number_input("Număr camere", min_value=0, max_value=20, value=3, step=1)
useful_surface = st.number_input("Suprafață utilă (mp)", min_value=1.0, value=50.0, step=10.0)
built_surface = st.number_input("Suprafață construită (mp)", min_value=0.0, value=70.0, step=10.0)
construction_year = st.number_input("An construcție", min_value=1700, max_value=2026, value=2010, step=1)
bathrooms_count = st.number_input("Număr băi", min_value=0, max_value=10, value=1, step=1)
level = st.number_input("Etaj", min_value=0, max_value=30, value=2, step=1)
max_level = st.number_input("Numarul de etaje ale blocului", min_value=0, max_value=200, value=8, step=1)
garages_count = st.number_input("Număr garaje", min_value=0, max_value=10, value=0, step=1)

# LOCATION
location_onehot_cols = [c for c in feature_columns if c.startswith("location_area_")]
location_options = ["Alege zona"] + [c.replace("location_area_", "") for c in location_onehot_cols]
picked_location = st.selectbox("Cartier / zonă", options=location_options)

if st.button("Estimează prețul"):
    # Build one-row input with EXACT training columns
    row = {col: 0 for col in feature_columns}

    # Fill numeric
    row["rooms_count"] = rooms_count
    row["useful_surface"] = useful_surface
    row["built_surface"] = built_surface
    row["construction_year"] = construction_year
    row["bathrooms_count"] = bathrooms_count
    row["level"] = level
    row["max_level"] = max_level
    row["garages_count"] = garages_count

    # Fill location one-hot if chosen (not base)
    if picked_location != "(baza / drop_first)":
        col_name = "location_area_" + picked_location
        if col_name in row:
            row[col_name] = 1

    X_input = pd.DataFrame([row], columns=feature_columns)

    # Scale ONLY numeric columns (same list ca în preprocessing)
    numeric_cols = [
        "rooms_count","useful_surface","built_surface","construction_year",
        "bathrooms_count","level","max_level","garages_count"
    ]
    X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

    # Predict price_log then convert back to price
    pred_log = model.predict(X_input)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"Preț estimat: {pred_price:,.0f}€")
    #st.caption("Model: Regresie liniară pe price_log; conversie înapoi cu expm1().")

"""
Rulare:
venv\Scripts\python -m streamlit run 5.interfata.py
"""