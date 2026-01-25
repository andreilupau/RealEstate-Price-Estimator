import pandas as pd
import os
import matplotlib.pyplot as plt #afisari
import numpy as np
#Incarcam din nou csv-ul
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data/house_offers.csv")
dataframe = pd.read_csv(csv_path)


### Curatare minima pentru grafice (nu e necesar cu acestea)
keep_cols = [
    "price","location_area","useful_surface","construction_year"
]
dataframe = dataframe[keep_cols].copy()

dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce")
dataframe["useful_surface"] = pd.to_numeric(dataframe["useful_surface"], errors="coerce")
dataframe["construction_year"] = pd.to_numeric(dataframe["construction_year"], errors="coerce")

dataframe = dataframe.dropna()
dataframe = dataframe[dataframe["price"] > 0]
mean_price_by_area = dataframe.groupby("location_area")["price"].mean().sort_values()


#=========================#
# 2. Analiza exploratorie |
#=========================#

# =========================
#  PREȚ MEDIU per SUPRAFAȚĂ(mp)
# =========================
# grupăm suprafața în intervale (bins)
dataframe["surface_bin"] = pd.cut(
    dataframe["useful_surface"],
    bins=np.arange(0, 201, 10)  # din 10 în 10 mp
)

avg_price_by_surface = (
    dataframe
    .groupby("surface_bin")["price"]
    .mean()
    / 1000  # mii €
)

plt.figure(figsize=(8,5))
avg_price_by_surface.plot(marker="o")

plt.xlabel("Suprafață utilă (intervale de 10 mp)")
plt.ylabel("Preț mediu (mii €)")
plt.title("Preț mediu în funcție de suprafața utilă in anul 2020")
plt.grid(True, alpha=0.3)
plt.show()


# =========================
#  PREȚ PE METRU PĂTRAT (€/mp) Cartiere
# =========================
# calculăm prețul pe mp
dataframe["price_per_mp"] = dataframe["price"] / dataframe["useful_surface"]

price_mp_by_area = (
    dataframe
    .groupby("location_area")["price_per_mp"]
    .median()
    .sort_values()
)

price_mp_by_area.tail(10).plot(kind="bar", figsize=(8,4))
plt.ylabel("Preț median (€/mp)")
plt.title("Top 10 cartiere după prețul pe metru pătrat")
plt.grid(True, axis="y", alpha=0.3)
plt.show()
