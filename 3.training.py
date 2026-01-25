import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

#importam datele
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data/bucuresti_ready.csv")
df = pd.read_csv(csv_path)

#=============================#
# 1. X / Y  +  TRAIN AND TEST
#=============================#
X = df.drop(columns=["price", "price_log"]) #le-am scos pe acestea din test (evident)
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


#==============================#
# 2. APLICAM MODELELE DE REGRESIE
#==============================#

#MODEL 1: Regresie liniară
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

#Evaluare/afisari
print("Regresie liniară")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

#MODEL 2: Arbore de decizie
tree = DecisionTreeRegressor(max_depth=6, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

#Evaluare/afisari
print("\nArbore de decizie")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))


#MODEL 3: Rețea neuronală
nn = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])
nn.compile(optimizer="adam", loss="mse")
nn.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
y_pred = nn.predict(X_test).flatten()

#Evaluare/afisari
print("\nRețea neuronală")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))


feature_columns = X.columns.tolist()
joblib.dump(lin_reg, "models/model_linreg.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
print("Saved: models/model_linreg.pkl, feature_columns.pkl")


#rulare manuala: venv\Scripts\python "3.training.py"