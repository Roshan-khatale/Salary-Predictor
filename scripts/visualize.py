import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/salary_data.csv")
X = data[["YearsExperience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()