# Bucharest Real Estate Price Estimator (ML + Streamlit)

A small end-to-end machine learning project that estimates apartment/house prices in Bucharest based on listing features (rooms, surface, year, floor, etc.). It includes data preprocessing, model training, a Streamlit UI for single predictions, and batch CSV predictions.

<img width="1919" height="915" alt="Screenshot 2026-01-25 135511" src="https://github.com/user-attachments/assets/3e18e021-15c5-4743-abb5-417fa741fb0e" />

## What problem does it solve?

Real estate prices vary a lot by size, location, and building characteristics. This project helps you:

- get a fast **price estimate in EUR** for a property given its features
- run **batch predictions** for multiple listings from a CSV
- do a quick **exploratory analysis** of the raw dataset

## Tech stack

- Python, pandas, NumPy
- scikit-learn (model + scaler)
- Streamlit (web UI)
- joblib (model persistence)
- Plotly (optional charts in the UI)

## Data

- Raw dataset: `data/house_offers.csv`
- Preprocessed dataset: `data/bucuresti_ready.csv`

> **Warning / Disclaimer**
>
> This project uses a dataset collected around **2020** to train the model, so the estimated prices will not be fully accurate in today’s marketplace.
> You _can_ use a more recent dataset, but it will require **code changes** (it’s not plug-and-play): you’ll need to update the preprocessing/training pipeline and then retrain the model.

## Project scripts

- `1.preprocessing.py` — preprocessing + feature engineering + scaler export
- `2.exploratory_analysis.py` — exploratory plots (matplotlib)
- `3.training.py` — training + evaluation + exporting model artifacts
- `4.interface.py` — Streamlit app (single prediction, batch prediction, data exploration)

## How to run (Windows)

### 1) Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the web interface

```bash
python -m streamlit run 4.interface.py
```

### That's it!

> Note: This repo already includes the prepared CSV in `data/` and the trained artifacts in `models/`, so you can run the UI directly.

### Optional: regenerate data + retrain

Run these only if you changed the dataset, preprocessing, or want to retrain the model.

```bash
python 1.preprocessing.py
python 3.training.py
```

### Optional: Run exploratory analysis

```bash
python 2.exploratory_analysis.py
```

## App features

- **Estimator**: single listing input → predicted price (EUR)
- **Batch predictions**: upload a CSV and download results
- **Data exploration**: quick stats + charts (uses `plotly` if installed)
- **Language toggle**: RO/EN UI strings (simple in-app i18n)

## Thank you for your attention!
