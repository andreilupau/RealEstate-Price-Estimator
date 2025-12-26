            # ==== 1. PREPROCESAREA DATELOR (Bucuresti Housing sep. 2020) ====

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# 1) Incarcare csv
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data/house_offers.csv")
dataframe = pd.read_csv(csv_path)

# =========================
# 2 Curatare. Eliminam coloanele pe care nu le vrem la antrenat
# =========================
keep_cols = [
    "price","location_area","rooms_count","useful_surface","built_surface","construction_year","bathrooms_count","level","max_level","garages_count"]

missing_cols = [c for c in keep_cols if c not in dataframe.columns]
if missing_cols:
    raise ValueError(f"Lipsesc coloanele: {missing_cols}. Verifică numele din CSV.")
dataframe = dataframe[keep_cols].copy()

#Afisari
#print("\n--- HEAD (raw, filtered cols) ---")
#print(dataframe.head())
#print("\n--- Missing values (raw) ---")
#print(dataframe.isnull().sum().sort_values(ascending=False))


            ### 2.1 Identifică valorile invalide
numeric_cols = [
    "price","rooms_count","useful_surface","built_surface","construction_year","bathrooms_count","level","max_level","garages_count"]

for col in numeric_cols:
    dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")



            ### 2.2 Tratarea valorilor lipsă
# location_area: dacă lipsește, punem "Unknown"
dataframe["location_area"] = dataframe["location_area"].fillna("Unknown").astype(str).str.strip()

# numeric: completăm cu mediană
for col in numeric_cols:
    if dataframe[col].isnull().any():
        dataframe[col] = dataframe[col].fillna(dataframe[col].median())

#print("\n--- Missing values (after fill) ---")
#print(dataframe.isnull().sum().sort_values(ascending=False))
#elimină rânduri cu preț invalid (0 sau negativ)
dataframe = dataframe[dataframe["price"] > 0].copy()


            ### 2.3 price_log ajută mult la regresie / NN
dataframe["price_log"] = np.log1p(dataframe["price"])
            ### 2.4 Transformă textul (zona) în numere
dataframe = pd.get_dummies(dataframe, columns=["location_area"], drop_first=True)


            ### 2.5 Scalare pt.ML
# Nu scalăm "price" (target) și nici "price_log" (target alternativ)
feature_cols_to_scale = [
    "rooms_count","useful_surface","built_surface","construction_year","bathrooms_count","level","max_level","garages_count",
]

scaler = StandardScaler()
dataframe[feature_cols_to_scale] = scaler.fit_transform(dataframe[feature_cols_to_scale])

#print("\n--- DUPĂ PREPROCESARE: head() ---")
#print(dataframe.head())
#print("\n--- DUPĂ PREPROCESARE: info() ---")
#print(dataframe.info())
#print("\n--- Missing values (final) ---")
#print(dataframe.isnull().sum().sort_values(ascending=False))


# =========================
# 3 SAVE READY CSV
# =========================

output_path = os.path.join(script_dir, "bucuresti_ready.csv")
dataframe.to_csv(output_path, index=False)
print(f"\nDataset preprocesat salvat în: {output_path}")

joblib.dump(scaler, "scaler.pkl")
print("Saved: scaler.pkl")