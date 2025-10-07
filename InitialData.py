# Import libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# Load data
df = pd.read_csv("Spotify_Dataset_V3.csv", sep = ";")
# Transform the semi-string columns to only numbers
df["# of Artist"] = df["# of Artist"].str.extract(r'(\d+)').astype(float)
df["# of Nationality"] = df["# of Nationality"].str.extract(r'(\d+)').astype(float)
# Transform the dates into only months - is there a correlation?
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
df["Month"] = df["Date"].dt.month
# Scout data's structure. Data is clean, no need for further cleaning
target_col = "Points (Total)"
string_cols = ["Rank", "Title", "Artists", "Date", "Artist (Ind.)", "Nationality", "Continent", "Points (Ind for each Artist/Nat)", "id", "Song URL", "# of Artist", "# of Nationality"]

y = df[target_col]
X = df.drop(columns=[target_col] + string_cols)
# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
feature_names = X.columns.to_list()
# --- Normalize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

df.head()