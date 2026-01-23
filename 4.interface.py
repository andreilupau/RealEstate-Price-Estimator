from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # plotly is optional; app still runs without charts
    px = None


# =========================
# Page config + styling (skip at line 377 for Data exploration)
# =========================
st.set_page_config(
    page_title="Imobiliare AI • Bucharest",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  /* Slightly tighter default spacing */
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

  /* Header card */
  .hero {
    border-radius: 16px;
    padding: 18px 18px;
    background: radial-gradient(1200px circle at 0% 0%, rgba(99,102,241,.28), transparent 50%),
                radial-gradient(800px circle at 100% 0%, rgba(16,185,129,.22), transparent 48%),
                linear-gradient(135deg, rgba(15,23,42,.92), rgba(2,6,23,.92));
    border: 1px solid rgba(148, 163, 184, 0.18);
  }
  .hero h1 { margin: 0; font-size: 1.8rem; color: #e2e8f0; }
  .hero p { margin: .35rem 0 0 0; color: rgba(226,232,240,.85); }

  /* Small pill */
  .pill {
    display:inline-block;
    padding: 2px 10px;
    margin-right: 8px;
    border-radius: 999px;
    font-size: .8rem;
    background: rgba(148,163,184,.12);
    border: 1px solid rgba(148,163,184,.18);
    color: rgba(226,232,240,.9);
  }

  /* Result card */
  .result {
    border-radius: 16px;
    padding: 16px;
    background: linear-gradient(135deg, rgba(2,6,23,.75), rgba(15,23,42,.55));
    border: 1px solid rgba(148, 163, 184, 0.18);
  }
  .result .big { font-size: 2rem; font-weight: 700; color: #e2e8f0; margin: 0; }
  .result .muted { color: rgba(226,232,240,.75); margin: .35rem 0 0 0; }

    /* Green success result variant */
    .result--success {
        background: linear-gradient(135deg, rgba(16,185,129,.35), rgba(16,185,129,.12));
        border: 1px solid rgba(16,185,129,.55);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# i18n (RO / EN)
# =========================
LANG_OPTIONS = {
    "Romanian": "ro",
    "English": "en",
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "ro"


I18N: dict[str, dict[str, str]] = {
    "ro": {
        "hero_title": "Imobiliare AI • Estimator de preț",
        "hero_subtitle": "Introdu detalii despre locuință și primești o estimare rapidă (în €). Include și predicții batch + explorare de date.",
        "nav": "Navigație",
        "language": "Limbă",
        "artifacts_loaded": "Artefacte încărcate",
        "page_estimator": "Estimator",
        "page_batch": "Predicții batch",
        "page_explore": "Explorare date",
        "page_about": "Despre",
        "input_subheader": "Introdu datele locuinței",
        "rooms_count": "Număr camere",
        "useful_surface": "Suprafață utilă (mp)",
        "built_surface": "Suprafață construită (mp)",
        "construction_year": "An construcție",
        "bathrooms_count": "Număr băi",
        "level": "Etaj",
        "max_level": "Etaje bloc",
        "garages_count": "Număr garaje",
        "area_caption": "Zonă / cartier (opțional)",
        "area_label": "Cartier / zonă",
        "area_none": "(fără selecție — baza drop_first)",
        "estimate": "Estimează prețul",
        "result": "Rezultat",
        "bad_levels": "Etajul este mai mare decât numărul de etaje ale blocului. Verifică valorile.",
        "approx_per_mp": "≈ {value} / mp (după suprafața utilă)",
        "metric_rooms": "Camere",
        "metric_useful": "Suprafață utilă",
        "metric_area": "Zonă",
        "area_base": "(baza drop_first)",
        "model_caption": "Modelul prezice `price_log`, apoi convertește înapoi cu `expm1()`.",
        "fill_form": "Completează formularul și apasă «Estimează prețul».",
        "batch_title": "Predicții batch (CSV upload)",
        "batch_desc": "Încarcă un CSV cu coloanele: `rooms_count`, `useful_surface`, `built_surface`, `construction_year`, `bathrooms_count`, `level`, `max_level`, `garages_count` și opțional `location_area`.",
        "download_sample": "Descarcă un exemplu CSV",
        "upload_csv": "CSV",
        "missing_cols": "Lipsesc coloanele obligatorii: {cols}",
        "download_preds": "Descarcă predicțiile",
        "explore_title": "Explorare rapidă (dataset brut)",
        "explore_caption": "Citește din `data/house_offers.csv` dacă există.",
        "missing_raw": "Nu găsesc `data/house_offers.csv`.",
        "metric_ads": "Anunțuri",
        "metric_median_price": "Preț median",
        "metric_median_eurmp": "€/mp median",
        "top_n": "Top N cartiere (€/mp median)",
        "chart_title": "Top {n} cartiere după €/mp (median)",
        "plotly_optional": "Instalează `plotly` pentru grafice interactive (opțional).",
        "preview": "Preview date",
        "about_title": "Despre",
        "about_text": "Aplicația folosește un model de regresie liniară antrenat pe dataset-ul preprocesat (`data/bucuresti_ready.csv`). Interfața aliniază input-ul la `feature_columns.pkl` și aplică același `StandardScaler` ca în preprocessing.",
        "tips": """\
**Tips**
- Dacă ai schimbat preprocessing-ul (coloane / dummies), re-rulează `1.preprocesare.py` și `3si4.antrenare&evaluare.py`.
- Pentru UI: `venv\\Scripts\\python -m streamlit run 5.interfata.py`.
""",
        "scaler_missing": "Nu găsesc scaler-ul. Aștept `models/scaler.pkl` sau `data/scaler.pkl`. Rulează `1.preprocesare.py` și/sau verifică fișierele din `models/`.",
    },
    "en": {
        "hero_title": "Real Estate AI • Price estimator",
        "hero_subtitle": "Enter property details and get a quick estimate (in €). Includes batch predictions and data exploration.",
        "nav": "Navigation",
        "language": "Language",
        "artifacts_loaded": "Loaded artifacts",
        "page_estimator": "Estimator",
        "page_batch": "Batch predictions",
        "page_explore": "Data exploration",
        "page_about": "About",
        "input_subheader": "Enter property details",
        "rooms_count": "Rooms",
        "useful_surface": "Usable area (sqm)",
        "built_surface": "Built area (sqm)",
        "construction_year": "Construction year",
        "bathrooms_count": "Bathrooms",
        "level": "Floor",
        "max_level": "Building floors",
        "garages_count": "Garages",
        "area_caption": "Area / neighborhood (optional)",
        "area_label": "Neighborhood / area",
        "area_none": "(no selection — drop_first baseline)",
        "estimate": "Estimate price",
        "result": "Result",
        "bad_levels": "Floor is higher than the building's maximum floor count. Please double-check the values.",
        "approx_per_mp": "≈ {value} / sqm (based on usable area)",
        "metric_rooms": "Rooms",
        "metric_useful": "Usable area",
        "metric_area": "Area",
        "area_base": "(drop_first baseline)",
        "model_caption": "The model predicts `price_log`, then converts back using `expm1()`.",
        "fill_form": "Fill in the form and click “Estimate price”.",
        "batch_title": "Batch predictions (CSV upload)",
        "batch_desc": "Upload a CSV with columns: `rooms_count`, `useful_surface`, `built_surface`, `construction_year`, `bathrooms_count`, `level`, `max_level`, `garages_count`, and optionally `location_area`.",
        "download_sample": "Download sample CSV",
        "upload_csv": "CSV",
        "missing_cols": "Missing required columns: {cols}",
        "download_preds": "Download predictions",
        "explore_title": "Quick exploration (raw dataset)",
        "explore_caption": "Reads from `data/house_offers.csv` if present.",
        "missing_raw": "Cannot find `data/house_offers.csv`.",
        "metric_ads": "Listings",
        "metric_median_price": "Median price",
        "metric_median_eurmp": "Median €/sqm",
        "top_n": "Top N neighborhoods (median €/sqm)",
        "chart_title": "Top {n} neighborhoods by median €/sqm",
        "plotly_optional": "Install `plotly` for interactive charts (optional).",
        "preview": "Data preview",
        "about_title": "About",
        "about_text": "This app uses a linear regression model trained on the preprocessed dataset (`data/bucuresti_ready.csv`). The UI aligns inputs to `feature_columns.pkl` and applies the same `StandardScaler` used during preprocessing.",
        "tips": """\
**Tips**
- If you changed preprocessing (columns / dummies), rerun `1.preprocesare.py` and `3si4.antrenare&evaluare.py`.
""",
        "scaler_missing": "Scaler not found. Expected `models/scaler.pkl` or `data/scaler.pkl`. Run `1.preprocesare.py` and/or verify files in `models/`.",
    },
}


def t(key: str, **kwargs: object) -> str:
    lang = st.session_state.get("lang", "ro")
    template = I18N.get(lang, I18N["ro"]).get(key, I18N["ro"].get(key, key))
    return template.format(**kwargs)


LANG = st.session_state.get("lang", "ro")


# =========================
# Domain constants
# =========================
NUMERIC_COLS: list[str] = [
    "rooms_count",
    "useful_surface",
    "built_surface",
    "construction_year",
    "bathrooms_count",
    "level",
    "max_level",
    "garages_count",
]


@dataclass(frozen=True)
class Artifacts:
    model: object
    scaler: object
    feature_columns: list[str]


def _format_eur(value: float, lang: str = "ro") -> str:
    if lang == "ro":
        # 123,456 -> 123.456 (RO-ish) + €
        s = f"{value:,.0f}"
        s = s.replace(",", " ").replace(" ", ".")
        return f"{s} €"
    return f"{value:,.0f} €"


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Artifacts:
    model = joblib.load("models/model_linreg.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    # Prefer models/scaler.pkl (as your UI expects), fallback to data/scaler.pkl (from preprocessing)
    scaler_path_candidates: Iterable[str] = ("models/scaler.pkl", "data/scaler.pkl")
    scaler = None
    for p in scaler_path_candidates:
        if os.path.exists(p):
            scaler = joblib.load(p)
            break
    if scaler is None:
        raise FileNotFoundError(t("scaler_missing"))

    return Artifacts(model=model, scaler=scaler, feature_columns=list(feature_columns))


def build_feature_row(
    *,
    artifacts: Artifacts,
    rooms_count: int,
    useful_surface: float,
    built_surface: float,
    construction_year: int,
    bathrooms_count: int,
    level: int,
    max_level: int,
    garages_count: int,
    location_area: str | None,
) -> pd.DataFrame:
    row = {col: 0 for col in artifacts.feature_columns}

    # raw numeric values; we scale them next
    row["rooms_count"] = rooms_count
    row["useful_surface"] = useful_surface
    row["built_surface"] = built_surface
    row["construction_year"] = construction_year
    row["bathrooms_count"] = bathrooms_count
    row["level"] = level
    row["max_level"] = max_level
    row["garages_count"] = garages_count

    if location_area:
        col_name = "location_area_" + location_area
        if col_name in row:
            row[col_name] = 1

    X_input = pd.DataFrame([row], columns=artifacts.feature_columns)
    X_input[NUMERIC_COLS] = artifacts.scaler.transform(X_input[NUMERIC_COLS])
    return X_input


def predict_price_eur(artifacts: Artifacts, X_input: pd.DataFrame) -> float:
    pred_log = float(artifacts.model.predict(X_input)[0])
    return float(np.expm1(pred_log))


artifacts = load_artifacts()

# Extract locations from training columns (drop_first=True => one category is the implicit base)
location_onehot_cols = [c for c in artifacts.feature_columns if c.startswith("location_area_")]
location_names = [c.replace("location_area_", "") for c in location_onehot_cols]


# =========================
# Header
# =========================
st.markdown(
        f"""
<div class="hero">
    <h1>{t('hero_title')}</h1>
    <p>{t('hero_subtitle')}</p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Sidebar navigation
# =========================
with st.sidebar:
    # Language toggle
    selected_lang_label = st.selectbox(
        "Language / Limbă",
        options=list(LANG_OPTIONS.keys()),
        index=list(LANG_OPTIONS.values()).index(LANG),
    )
    selected_lang = LANG_OPTIONS[selected_lang_label]
    if selected_lang != LANG:
        st.session_state["lang"] = selected_lang
        st.rerun()

    st.subheader(t("nav"))

    PAGES = {
        "estimator": t("page_estimator"),
        "batch": t("page_batch"),
        "explore": t("page_explore"),
        "about": t("page_about"),
    }

    page = st.radio(
        "",
        options=list(PAGES.keys()),
        index=0,
        format_func=lambda k: PAGES[k],
    )

    st.divider()
    st.caption(t("artifacts_loaded"))
    st.write("- `models/model_linreg.pkl`")
    st.write("- `models/feature_columns.pkl`")
    st.write("- scaler: `models/scaler.pkl` (fallback `data/scaler.pkl`)")


# =========================
# Page: Estimator
# =========================
if page == "estimator":
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader(t("input_subheader"))
        with st.form("predict_form", border=True):
            c1, c2 = st.columns(2)

            with c1:
                rooms_count = st.number_input(t("rooms_count"), min_value=0, max_value=20, value=3, step=1)
                useful_surface = st.number_input(t("useful_surface"), min_value=1.0, value=50.0, step=10.0)
                construction_year = st.number_input(
                    t("construction_year"),
                    min_value=1700,
                    max_value=2026,
                    value=2010,
                    step=1,
                )
                bathrooms_count = st.number_input(t("bathrooms_count"), min_value=0, max_value=10, value=1, step=1)

            with c2:
                built_surface = st.number_input(t("built_surface"), min_value=0.0, value=70.0, step=10.0)
                level = st.number_input(t("level"), min_value=0, max_value=60, value=2, step=1)
                max_level = st.number_input(t("max_level"), min_value=0, max_value=200, value=8, step=1)
                garages_count = st.number_input(t("garages_count"), min_value=0, max_value=10, value=0, step=1)

            st.divider()
            st.caption(t("area_caption"))
            location_choice = st.selectbox(
                t("area_label"),
                options=[t("area_none")] + sorted(location_names),
                index=0,
            )

            submitted = st.form_submit_button(t("estimate"), use_container_width=True)

    with right:
        st.subheader(t("result"))
        if submitted:
            location_area = None if location_choice.startswith("(") else location_choice

            if max_level > 0 and level > max_level:
                st.warning(t("bad_levels"))

            X_input = build_feature_row(
                artifacts=artifacts,
                rooms_count=int(rooms_count),
                useful_surface=float(useful_surface),
                built_surface=float(built_surface),
                construction_year=int(construction_year),
                bathrooms_count=int(bathrooms_count),
                level=int(level),
                max_level=int(max_level),
                garages_count=int(garages_count),
                location_area=location_area,
            )
            price = predict_price_eur(artifacts, X_input)
            price_per_mp = price / max(float(useful_surface), 1.0)

            st.markdown(
                f"""
<div class="result result--success">
    <p class="big">{_format_eur(price, LANG)}</p>
    <p class="muted">{t('approx_per_mp', value=_format_eur(price_per_mp, LANG))}</p>
</div>
""",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric(t("metric_rooms"), int(rooms_count))
            m2.metric(t("metric_useful"), f"{float(useful_surface):.0f} mp")
            m3.metric(t("metric_area"), location_area or t("area_base"))

            st.caption(t("model_caption"))
        else:
            st.info(t("fill_form"), icon="ℹ️")


# =========================
# Page: Batch predictions
# =========================
elif page == "batch":
    st.subheader(t("batch_title"))
    st.write(t("batch_desc"))

    sample = pd.DataFrame(
        [
            {
                "rooms_count": 2,
                "useful_surface": 52,
                "built_surface": 70,
                "construction_year": 2012,
                "bathrooms_count": 1,
                "level": 3,
                "max_level": 8,
                "garages_count": 0,
                "location_area": "Titan",
            }
        ]
    )
    st.download_button(
        t("download_sample"),
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_pred.csv" if LANG == "en" else "exemplu_pred.csv",
        mime="text/csv",
        use_container_width=True,
    )

    upload = st.file_uploader(t("upload_csv"), type=["csv"], accept_multiple_files=False)
    if upload is not None:
        df = pd.read_csv(upload)
        missing = [c for c in NUMERIC_COLS if c not in df.columns]
        if missing:
            st.error(t("missing_cols", cols=missing))
            st.stop()

        df = df.copy()
        if "location_area" not in df.columns:
            df["location_area"] = None

        # Build feature matrix aligned to training columns
        rows = []
        for _, r in df.iterrows():
            loc = r.get("location_area")
            loc = None if pd.isna(loc) else str(loc).strip()
            rows.append(
                build_feature_row(
                    artifacts=artifacts,
                    rooms_count=int(r["rooms_count"]),
                    useful_surface=float(r["useful_surface"]),
                    built_surface=float(r["built_surface"]),
                    construction_year=int(r["construction_year"]),
                    bathrooms_count=int(r["bathrooms_count"]),
                    level=int(r["level"]),
                    max_level=int(r["max_level"]),
                    garages_count=int(r["garages_count"]),
                    location_area=loc,
                )
            )
        X = pd.concat(rows, ignore_index=True)
        preds = artifacts.model.predict(X)
        df_out = df.copy()
        df_out["pred_price_eur"] = np.expm1(preds)
        st.dataframe(df_out, use_container_width=True, hide_index=True)

        st.download_button(
            t("download_preds"),
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv" if LANG == "en" else "predictii.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================
# Page: Data exploration
# =========================
elif page == "explore":
    st.subheader(t("explore_title"))
    st.caption(t("explore_caption"))

    df_raw = _safe_read_csv("data/house_offers.csv")
    if df_raw is None:
        st.warning(t("missing_raw"), icon="⚠️")
        st.stop()

    keep = ["price", "location_area", "useful_surface", "construction_year"]
    dfv = df_raw[keep].copy()
    dfv["price"] = pd.to_numeric(dfv["price"], errors="coerce")
    dfv["useful_surface"] = pd.to_numeric(dfv["useful_surface"], errors="coerce")
    dfv["construction_year"] = pd.to_numeric(dfv["construction_year"], errors="coerce")
    dfv["location_area"] = dfv["location_area"].astype(str).str.strip()
    dfv = dfv.dropna()
    dfv = dfv[(dfv["price"] > 0) & (dfv["useful_surface"] > 0)]
    dfv["price_per_mp"] = dfv["price"] / dfv["useful_surface"]

    c1, c2, c3 = st.columns(3)
    ads_value = f"{len(dfv):,}".replace(",", ".") if LANG == "ro" else f"{len(dfv):,}"
    c1.metric(t("metric_ads"), ads_value)
    c2.metric(t("metric_median_price"), _format_eur(float(dfv["price"].median()), LANG))
    c3.metric(t("metric_median_eurmp"), _format_eur(float(dfv["price_per_mp"].median()), LANG))

    st.divider()
    top_n = st.slider(t("top_n"), min_value=5, max_value=30, value=10)
    by_area = (
        dfv.groupby("location_area", as_index=False)
        .agg(median_eur_mp=("price_per_mp", "median"), count=("price", "size"))
        .sort_values("median_eur_mp", ascending=False)
        .head(top_n)
    )

    if px is not None:
        fig = px.bar(
            by_area.sort_values("median_eur_mp", ascending=True),
            x="median_eur_mp",
            y="location_area",
            orientation="h",
            title=t("chart_title", n=top_n),
            hover_data={"count": True, "median_eur_mp": ":.0f"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("plotly_optional"))
        st.dataframe(by_area, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(t("preview"))
    st.dataframe(dfv.sample(min(200, len(dfv))), use_container_width=True, hide_index=True)


# =========================
# Page: About
# =========================
else:
    st.subheader(t("about_title"))
    st.write(t("about_text"))
    st.markdown(t("tips"))


#to run the ui type in console: venv\\Scripts\\python -m streamlit run 5.interfata.py