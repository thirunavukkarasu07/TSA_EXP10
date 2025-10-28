# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 28.10.2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- Load the dataset ---
data = pd.read_csv('housing_price_dataset.csv')

# --- Prepare data: group by year and average prices ---
data_yearly = data.groupby('YearBuilt')['Price'].mean().sort_index()

# Convert YearBuilt to datetime index
data_yearly.index = pd.to_datetime(data_yearly.index, format='%Y')
data_yearly = pd.DataFrame(data_yearly)
data_yearly.columns = ['AveragePrice']

# --- Plot the time series ---
plt.figure(figsize=(10, 5))
plt.plot(data_yearly.index, data_yearly['AveragePrice'], color='blue')
plt.xlabel('Year')
plt.ylabel('Average House Price')
plt.title('Yearly Average House Price')
plt.grid(True)
plt.show()

# --- Function to check stationarity ---
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] <= 0.05:
        print("✅ The series is stationary.")
    else:
        print("❗ The series is NOT stationary. Consider differencing or transformation.")

# --- Stationarity test ---
check_stationarity(data_yearly['AveragePrice'])

# --- Plot ACF and PACF ---
plt.figure(figsize=(8, 4))
plot_acf(data_yearly['AveragePrice'], lags=15)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(8, 4))
plot_pacf(data_yearly['AveragePrice'], lags=15)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# --- Split into train/test sets ---
train_size = int(len(data_yearly) * 0.8)
train, test = data_yearly['AveragePrice'][:train_size], data_yearly['AveragePrice'][train_size:]

# --- Fit SARIMA model ---
# Using a basic configuration for yearly data (can be tuned)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 10))
sarima_result = sarima_model.fit(disp=False)

# --- Forecast future prices for test period ---
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# --- Calculate RMSE ---
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.2f}')

# --- Plot actual vs predicted ---
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Training Data', color='blue')
plt.plot(test.index, test, label='Actual Prices (Test Data)', color='green')
plt.plot(test.index, predictions, label='Predicted Prices (SARIMA)', color='red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Average House Price')
plt.title('SARIMA Model Forecast of House Prices')
plt.legend()
plt.grid(True)
plt.show()

# --- Forecast for next 10 years ---
future_steps = 10
future_forecast = sarima_result.forecast(steps=future_steps)
print("\nForecasted Average Prices for the Next 10 Years:")
print(future_forecast)

# --- Plot future forecast ---
plt.figure(figsize=(10, 5))
plt.plot(data_yearly.index, data_yearly['AveragePrice'], label='Historical Data', color='blue')
plt.plot(future_forecast.index, future_forecast, label='Future Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Average House Price')
plt.title('Forecasted Future House Prices (SARIMA)')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT:

<img width="923" height="604" alt="image" src="https://github.com/user-attachments/assets/e90f9119-2d86-4a8f-9b99-f2ee294337d1" />

<img width="608" height="468" alt="image" src="https://github.com/user-attachments/assets/6220e2d9-397e-4dc0-99b3-dea84ddbac40" />

<img width="599" height="454" alt="image" src="https://github.com/user-attachments/assets/eb7a4875-1c67-4dac-807e-0533256c023a" />


<img width="894" height="514" alt="image" src="https://github.com/user-attachments/assets/4e700bdd-a56e-4de7-b6af-c7109f3c693e" />

<img width="907" height="502" alt="image" src="https://github.com/user-attachments/assets/55553c44-270c-4b78-8e96-0c6fa60f7c31" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
