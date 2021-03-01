import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("data/salary_data.csv")
X = data[["YearsExperience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "models/salary_model.pkl")
print("Model trained and saved.")