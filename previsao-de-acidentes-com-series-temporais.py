# %% [markdown]
# # Task
# Perform a daily forecast of traffic accidents using the 'Acidentes' column as the target variable, employing an ARIMA or SARIMA model. The input data is in the "BASE.XLSX" file and contains the columns 'Data', 'Acidentes', 'Feridos', and 'Mortos'. The process should involve loading the data, preprocessing it (including handling missing dates and sorting), selecting the target variable, checking for stationarity and applying differencing if necessary, analyzing ACF and PACF plots to estimate model parameters, splitting the data into training and testing sets (using the last 20% for testing), training and evaluating ARIMA/SARIMA models with different parameter combinations using MAE, RMSE, and MAPE, analyzing model residuals (plotting and Ljung-Box test), forecasting for the next 30 days, and visualizing the forecast with a confidence interval. The expected libraries are pandas, statsmodels, matplotlib, seaborn, and sklearn.metrics.

# %% [markdown]
# ## Load data
# 
# ### Subtask:
# Load the data from the `Base.xlsx` file into a pandas DataFrame.
# 

# %% [markdown]
# **Reasoning**:
# Import pandas and load the Excel file into a DataFrame.
# 
# 

# %%
import pandas as pd

df = pd.read_excel('Base.xlsx')

# %% [markdown]
# **Reasoning**:
# Display the first few rows of the DataFrame to verify the data loading.
# 
# 

# %%
df.head()

# %% [markdown]
# ## Data preprocessing
# 
# ### Subtask:
# Convert the 'Data' column to datetime objects, set it as the index, and fill in any missing dates with a value of 0 for 'Acidentes', 'Feridos', and 'Mortos'. Then, sort the DataFrame by date.
# 

# %% [markdown]
# **Reasoning**:
# Convert the 'Data' column to datetime, set it as the index, fill missing dates with 0, and sort the index as instructed.
# 
# 

# %%
df['Data'] = pd.to_datetime(df['Data'])
df = df.set_index('Data')
df = df.resample('D').fillna(0)
df = df.sort_index()

# %% [markdown]
# **Reasoning**:
# The previous code failed because `fillna` on a resampled object expects a method (like 'ffill' or 'bfill'), not a value like 0. To fill with a specific value after resampling, use the `asfreq` method with the desired fill value.
# 
# 

# %%
df['Data'] = pd.to_datetime(df['Data'])
df = df.set_index('Data')
df = df.resample('D').asfreq(fill_value=0)
df = df.sort_index()

# %% [markdown]
# **Reasoning**:
# The error indicates that the 'Data' column is no longer present in the DataFrame, likely because the index was set in a previous, failed attempt within the same cell. The `code_block` command should contain the complete code for this step.
# 
# 

# %%
df['Data'] = pd.to_datetime(df['Data'])
df = df.set_index('Data')
df = df.resample('D').asfreq(fill_value=0)
df = df.sort_index()

# %% [markdown]
# **Reasoning**:
# The error "KeyError: 'Data'" persists because the DataFrame `df` is already indexed by 'Data' from a previous execution in the same kernel session. The code should start with the DataFrame *before* setting the index. I need to reload the data to ensure the 'Data' column exists before attempting to set it as the index.
# 
# 

# %%
import pandas as pd

df = pd.read_excel('Base.xlsx')
df['Data'] = pd.to_datetime(df['Data'])
df = df.set_index('Data')
df = df.resample('D').asfreq(fill_value=0)
df = df.sort_index()

# %% [markdown]
# ## Select target variable
# 
# ### Subtask:
# Isolate the 'Acidentes' column for time series modeling.
# 

# %% [markdown]
# **Reasoning**:
# Select the 'Acidentes' column from the DataFrame and store it in a new variable.
# 
# 

# %%
ts_data = df['Acidentes']

# %% [markdown]
# ## Check for stationarity
# 
# ### Subtask:
# Perform the Augmented Dickey-Fuller (ADF) test on the 'Acidentes' series to check for stationarity.
# 

# %% [markdown]
# **Reasoning**:
# Perform the Augmented Dickey-Fuller test on the time series data to check for stationarity and print the results.
# 
# 

# %%
from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(ts_data)
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])
print('Number of lags used: %i' % adf_test[2])
print('Critical Values:')
for key, value in adf_test[4].items():
    print('\t%s: %.3f' % (key, value))

# %% [markdown]
# ## Apply differencing (if needed)
# 
# ### Subtask:
# If the series is not stationary, apply differencing to make it stationary.
# 

# %% [markdown]
# **Reasoning**:
# Check the p-value from the ADF test result and apply differencing if it indicates non-stationarity.
# 
# 

# %%
p_value = adf_test[1]
if p_value > 0.05:
    ts_data_diff = ts_data.diff().dropna()
    print("Original data is non-stationary. Differencing applied.")
    print("Head of differenced data:")
    display(ts_data_diff.head())
else:
    print("Original data is stationary.")

# %% [markdown]
# ## Analyze acf and pacf
# 
# ### Subtask:
# Generate plots of the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to help determine the parameters (p, d, q) for the ARIMA model.
# 

# %% [markdown]
# **Reasoning**:
# Generate ACF and PACF plots for the differenced time series to identify potential parameters for the ARIMA model.
# 
# 

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

plot_acf(ts_data_diff, ax=axes[0])
plot_pacf(ts_data_diff, ax=axes[1])

plt.show()

# %% [markdown]
# ## Split data
# 
# ### Subtask:
# Separate the time series data into training and testing sets. Use the last 20% of the data for testing.
# 

# %% [markdown]
# **Reasoning**:
# Calculate the test set size, determine the split index, and create the training and testing sets. Then, print the lengths of the resulting sets.
# 
# 

# %%
# Define the split date
split_date = '2017-01-01'

train_data = ts_data[:split_date]
test_data = ts_data[split_date:]

print(f"Length of training data: {len(train_data)}")
print(f"Length of testing data: {len(test_data)}")

# %% [markdown]
# ## Train arima/sarima model
# 
# ### Subtask:
# Train an ARIMA or SARIMA model using the training data. Iterate through different combinations of parameters (p, d, q) to find the best fit.
# 

# %% [markdown]
# **Reasoning**:
# Import the necessary class for ARIMA modeling and define the parameter grid and variables for model selection.
# 
# 

# %%
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Define the parameter combinations to explore (p, d, q)
p_values = range(0, 3)
d_values = range(0, 2) # We know d=1 is likely appropriate, but include d=0 to explore
q_values = range(0, 3)

best_aic = float('inf')
best_order = None
best_model = None

# %% [markdown]
# **Reasoning**:
# Iterate through the defined parameter combinations, fit the ARIMA model, calculate AIC, and update the best model if a lower AIC is found.
# 
# 

# %%
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                model = ARIMA(train_data, order=order)
                model_fit = model.fit()
                aic = model_fit.aic

                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model = model_fit
                print(f'ARIMA{order} AIC={aic:.2f}')
            except Exception as e:
                print(f'ARIMA{order} failed: {e}')

print(f'\nBest ARIMA order: {best_order} with AIC={best_aic:.2f}')

# %% [markdown]
# ## Evaluate model
# 
# ### Subtask:
# Evaluate the trained model's performance on the test set using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
# 

# %% [markdown]
# **Reasoning**:
# Generate predictions on the test data, invert differencing if necessary, calculate evaluation metrics (MAE, RMSE, MAPE), and print the results.
# 
# 

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Generate predictions
# The start and end dates for prediction should match the test_data index
start_index = test_data.index[0]
end_index = test_data.index[-1]

predictions = best_model.predict(start=start_index, end=end_index)

# 2. Invert differencing if d=1
if best_order[1] == 1:
    # The predict method of ARIMA with d=1 already returns predictions on the original scale
    # This is because the model is fitted on the differenced data, but the predict method
    # handles the integration step automatically.
    # Therefore, no manual differencing inversion is needed for statsmodels ARIMA predict.
    print("Model was trained with d=1. Predictions are on the original scale.")
else:
    print("Model was trained with d=0. Predictions are on the original scale.")

# 3. Display Actual vs. Predicted for the test set
comparison_df = pd.DataFrame({'Actual': test_data, 'Predicted': predictions})
print("\nActual vs. Predicted on Test Set:")
display(comparison_df.head())


# 4. Calculate MAE
mae = mean_absolute_error(test_data, predictions)

# 5. Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))

# 6. Calculate MAPE
# Add a small epsilon to avoid division by zero
epsilon = 1e-10
mape = np.mean(np.abs((test_data - predictions) / (test_data + epsilon))) * 100

# 7. Print metrics
print(f'\nMean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# %% [markdown]
# ## Residual analysis
# 
# ### Subtask:
# Analyze the residuals of the model by plotting them and performing the Ljung-Box test to check for autocorrelation in the residuals.
# 

# %% [markdown]
# **Reasoning**:
# Analyze the residuals by plotting them and performing the Ljung-Box test.
# 
# 

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# 1. Get the residuals
residuals = best_model.resid

# 2. Plot the residuals over time
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals over Time')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.grid(True)
plt.show()

# 3. Perform the Ljung-Box test
# Consider using a few different lag values (e.g., 10, 20)
ljungbox_test_10 = acorr_ljungbox(residuals, lags=[10])
ljungbox_test_20 = acorr_ljungbox(residuals, lags=[20])


# 4. Print the results of the Ljung-Box test
print("\nLjung-Box Test Results (Lags=10):")
display(ljungbox_test_10)

print("\nLjung-Box Test Results (Lags=20):")
display(ljungbox_test_20)

# %% [markdown]
# ## Forecast future values
# 
# ### Subtask:
# Use the trained model to forecast the number of accidents for the next 30 days.
# 

# %% [markdown]
# **Reasoning**:
# Use the trained model to forecast the number of accidents for the next 30 days.
# 
# 

# %%
# Determine the start date for the forecast (day after the last date in the test set)
last_test_date = test_data.index[-1]
forecast_start_date = last_test_date + pd.Timedelta(days=1)

# Determine the end date for the forecast (30 days after the start date)
forecast_end_date = forecast_start_date + pd.Timedelta(days=29)

# Generate the forecast for the next 30 days
forecast = best_model.predict(start=forecast_start_date, end=forecast_end_date)

# Print the generated forecast values
print("\nForecast for the next 30 days:")
print(forecast)

# %% [markdown]
# ## Visualize forecast
# 
# ### Subtask:
# Display a plot of the historical data, the test set, and the forecast with a confidence interval.
# 

# %% [markdown]
# **Reasoning**:
# Generate the plot showing the historical data, test data, forecast, and confidence intervals.
# 
# 

# %%
import matplotlib.pyplot as plt

# Get the confidence intervals
forecast_result = best_model.get_forecast(steps=30)
confidence_interval = forecast_result.conf_int()

# Plot the historical data (training and testing)
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Training Data', color='blue')
plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')

# Plot the forecast
plt.plot(forecast.index, forecast, label='Forecast', color='red')

# Plot the confidence interval
plt.fill_between(confidence_interval.index,
                 confidence_interval.iloc[:, 0],
                 confidence_interval.iloc[:, 1], color='pink', alpha=.3, label='Confidence Interval')

# Add titles and labels
plt.title('Traffic Accident Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# *   The initial ADF test on the 'Acidentes' time series indicated non-stationarity (p-value 0.680).
# *   First-order differencing was applied to the time series to address non-stationarity.
# *   Analysis of ACF and PACF plots of the differenced data was performed to aid in selecting ARIMA parameters.
# *   The data was split, with 854 days for training and 213 days for testing.
# *   An ARIMA(2, 1, 1) model was identified as the best-fitting model based on the lowest AIC (approximately 10532.46) after iterating through various parameter combinations.
# *   Model evaluation on the test set yielded the following metrics: MAE of 99.23, RMSE of 153.23, and an extremely high MAPE of 64780912243064.05%.
# *   The Ljung-Box test on the model residuals indicated significant autocorrelation (p-values < 0.05 for lags 10 and 20), suggesting the model has not fully captured the time series patterns.
# *   The 30-day forecast showed a constant predicted value of approximately 69.5 for the entire forecast period.
# *   A visualization comparing the historical data, actual test data, and the 30-day forecast with its confidence interval was generated.
# 
# ### Insights or Next Steps
# 
# *   The high MAPE and significant autocorrelation in residuals suggest the ARIMA(2, 1, 1) model may not be adequate. Exploring SARIMA models to capture potential seasonality in the daily accident data is a crucial next step.
# *   Investigating the periods with very low actual values in the test set might explain the extremely high MAPE and inform data transformation or model selection strategies.
# 

# %% [markdown]
# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# * The initial ADF test on the 'Acidentes' time series indicated non-stationarity (p-value 0.680).
# * First-order differencing was applied to the time series to address non-stationarity.
# * Analysis of ACF and PACF plots of the differenced data was performed to aid in selecting ARIMA parameters.
# * The data was split, with 854 days for training and 213 days for testing.
# * An ARIMA(2, 1, 1) model was identified as the best-fitting model based on the lowest AIC (approximately 10532.46) after iterating through various parameter combinations.
# * Model evaluation on the test set yielded the following metrics: MAE of {mae:.2f}, RMSE of {rmse:.2f}, and an extremely high MAPE of {mape:.2f}%.
# * The Ljung-Box test on the model residuals indicated significant autocorrelation (p-values < 0.05 for lags 10 and 20), suggesting the model has not fully captured the time series patterns.
# * The 30-day forecast showed a constant predicted value of approximately 69.5 for the entire forecast period.
# * A visualization comparing the historical data, actual test data, and the 30-day forecast with its confidence interval was generated.
# 
# ### Insights or Next Steps
# 
# * The high MAPE and significant autocorrelation in residuals suggest the ARIMA(2, 1, 1) model may not be adequate. Exploring SARIMA models to capture potential seasonality in the daily accident data is a crucial next step.
# * Investigating the periods with very low actual values in the test set might explain the extremely high MAPE and inform data transformation or model selection strategies.

# %% [markdown]
# ## Actual vs. Predicted Comparison on Test Set
# 
# ### Subtask:
# Visualize the actual 'Acidentes' values from the test set against the model's predictions for the same period.

# %% [markdown]
# **Reasoning**:
# Generate a plot to visually compare the actual values from the test set with the predicted values from the trained ARIMA model.

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
plt.plot(predictions.index, predictions, label='Predicted Test Data', color='red', linestyle='--')

plt.title('Actual vs. Predicted Traffic Accidents on Test Set')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Train arima/sarima model
# 
# ### Subtask:
# Train an ARIMA or SARIMA model using the training data. Iterate through different combinations of parameters (p, d, q) to find the best fit.

# %% [markdown]
# **Reasoning**:
# Import the necessary class for ARIMA modeling and define the parameter grid and variables for model selection.

# %%
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Define the parameter combinations to explore (p, d, q)
p_values = range(0, 3)
d_values = range(0, 2) # We know d=1 is likely appropriate, but include d=0 to explore
q_values = range(0, 3)

best_aic = float('inf')
best_order = None
best_model = None

# %% [markdown]
# **Reasoning**:
# Iterate through the defined parameter combinations, fit the ARIMA model, calculate AIC, and update the best model if a lower AIC is found.

# %%
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                model = ARIMA(train_data, order=order)
                model_fit = model.fit()
                aic = model_fit.aic

                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model = model_fit
                print(f'ARIMA{order} AIC={aic:.2f}')
            except Exception as e:
                print(f'ARIMA{order} failed: {e}')

print(f'\nBest ARIMA order: {best_order} with AIC={best_aic:.2f}')

# %% [markdown]
# ## Evaluate model
# 
# ### Subtask:
# Evaluate the trained model's performance on the test set using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

# %% [markdown]
# **Reasoning**:
# Generate predictions on the test data, invert differencing if necessary, calculate evaluation metrics (MAE, RMSE, MAPE), and print the results.

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Generate predictions
# The start and end dates for prediction should match the test_data index
start_index = test_data.index[0]
end_index = test_data.index[-1]

predictions = best_model.predict(start=start_index, end=end_index)

# 2. Invert differencing if d=1
if best_order[1] == 1:
    # The predict method of ARIMA with d=1 already returns predictions on the original scale
    # This is because the model is fitted on the differenced data, but the predict method
    # handles the integration step automatically.
    # Therefore, no manual differencing inversion is needed for statsmodels ARIMA predict.
    print("Model was trained with d=1. Predictions are on the original scale.")
else:
    print("Model was trained with d=0. Predictions are on the original scale.")

# 3. Display Actual vs. Predicted for the test set
comparison_df = pd.DataFrame({'Actual': test_data, 'Predicted': predictions})
print("\nActual vs. Predicted on Test Set:")
display(comparison_df.head())
display(comparison_df)


# 4. Calculate MAE
mae = mean_absolute_error(test_data, predictions)

# 5. Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))

# 6. Calculate MAPE
# Add a small epsilon to avoid division by zero
epsilon = 1e-10
mape = np.mean(np.abs((test_data - predictions) / (test_data + epsilon))) * 100

# 7. Print metrics
print(f'\nMean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# %% [markdown]
# ## Residual analysis
# 
# ### Subtask:
# Analyze the residuals of the model by plotting them and performing the Ljung-Box test to check for autocorrelation in the residuals.

# %% [markdown]
# **Reasoning**:
# Analyze the residuals by plotting them and performing the Ljung-Box test.

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# 1. Get the residuals
residuals = best_model.resid

# 2. Plot the residuals over time
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals over Time')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.grid(True)
plt.show()

# 3. Perform the Ljung-Box test
# Consider using a few different lag values (e.g., 10, 20)
ljungbox_test_10 = acorr_ljungbox(residuals, lags=[10])
ljungbox_test_20 = acorr_ljungbox(residuals, lags=[20])


# 4. Print the results of the Ljung-Box test
print("\nLjung-Box Test Results (Lags=10):")
display(ljungbox_test_10)

print("\nLjung-Box Test Results (Lags=20):")
display(ljungbox_test_20)

# %% [markdown]
# ## Forecast future values
# 
# ### Subtask:
# Use the trained model to forecast the number of accidents for the next 30 days.

# %% [markdown]
# **Reasoning**:
# Use the trained model to forecast the number of accidents for the next 30 days.

# %%
# Determine the start date for the forecast (day after the last date in the test set)
last_test_date = test_data.index[-1]
forecast_start_date = last_test_date + pd.Timedelta(days=1)

# Determine the end date for the forecast (30 days after the start date)
forecast_end_date = forecast_start_date + pd.Timedelta(days=29)

# Generate the forecast for the next 30 days
forecast = best_model.predict(start=forecast_start_date, end=forecast_end_date)

# Print the generated forecast values
print("\nForecast for the next 30 days:")
print(forecast)

# %% [markdown]
# ## Visualize forecast
# 
# ### Subtask:
# Display a plot of the historical data, the test set, and the forecast with a confidence interval.

# %% [markdown]
# **Reasoning**:
# Generate the plot showing the historical data, test data, forecast, and confidence intervals.

# %%
import matplotlib.pyplot as plt

# Get the confidence intervals
forecast_result = best_model.get_forecast(steps=30)
confidence_interval = forecast_result.conf_int()

# Plot the historical data (training and testing)
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Training Data', color='blue')
plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')

# Plot the forecast
plt.plot(forecast.index, forecast, label='Forecast', color='red')

# Plot the confidence interval
plt.fill_between(confidence_interval.index,
                 confidence_interval.iloc[:, 0],
                 confidence_interval.iloc[:, 1], color='pink', alpha=.3, label='Confidence Interval')

# Add titles and labels
plt.title('Traffic Accident Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.legend()
plt.grid(True)
plt.show()


