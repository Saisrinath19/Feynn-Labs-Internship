import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import zscore
from scipy.optimize import linprog

# Load dataset
data = pd.read_csv("energy_data.csv")

# Data Cleaning
data.dropna(inplace=True)
data = data[data['consumption'] > 0]

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Descriptive Statistics
print("Mean Consumption:", data['consumption'].mean())
print("Standard Deviation:", data['consumption'].std())
print(data[['temperature', 'consumption']].corr())

# Visualization
plt.figure(figsize=(10, 5))
sns.histplot(data['consumption'], bins=30, kde=True, color='blue')
plt.title("Energy Consumption Distribution")
plt.xlabel("Consumption (kWh)")
plt.ylabel("Frequency")
plt.show()

# Predictive Modeling (Time Series Forecasting with ARIMA)
model = ARIMA(data['consumption'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# Regression Model for Energy Consumption Prediction
X = data[['temperature', 'humidity']]
y = data['consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-Squared Score:", r2_score(y_test, y_pred))

# Optimization Using Linear Programming
costs = [0.12, 0.15, 0.20]
constraints = [[1, 1, 1]]
bounds = [(0, 5), (0, 3), (0, 4)]
result = linprog(costs, A_eq=constraints, b_eq=[10], bounds=bounds)
print("Optimal Schedule:", result.x)

# Anomaly Detection
z_scores = zscore(data['consumption'])
anomalies = data[np.abs(z_scores) > 3]
print("Anomalies Detected:", anomalies)

# Real-Time Monitoring and Alerts
def check_abnormal_consumption(consumption, threshold=1000):
    if consumption > threshold:
        print("Alert: High energy consumption detected!")

check_abnormal_consumption(1500)
