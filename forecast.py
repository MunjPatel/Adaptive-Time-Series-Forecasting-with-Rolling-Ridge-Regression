import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# importing data from and performing boolean-indexing for date filtering
ticker = yf.Ticker('^GSPC')
usa_data = ticker.history(period='max')
usa_data = usa_data[(usa_data.index >= "01-01-2014")&(usa_data.index < "01-01-2024")]

# setting parameters
window_size = 126
alpha = 1.0

# creating lagged features
usa_data['Lagged_Close'] = usa_data['Close'].shift(1)
usa_data['Lagged_Volume'] = usa_data['Volume'].shift(1)
usa_data = usa_data.dropna()

# rolling window predictions
predicted_prices = []
actual_prices = usa_data['Close'][window_size:].tolist()

X = usa_data[['Lagged_Close', 'Lagged_Volume']].values
y = usa_data['Close'].values

for i in tqdm(range(window_size, len(usa_data) - 1)):
    X_train = X[i-window_size:i]
    y_train = y[i-window_size:i]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    residuals = y_train - model.predict(X_train)
    mu = residuals.mean()
    sigma = residuals.std()

    X_test = X[i + 1].reshape(1, -1)
    predicted_price = model.predict(X_test)[0] + mu + sigma * np.random.normal()
    predicted_prices.append(predicted_price)

# aligning the predicted prices with the actual prices for plotting
actual_prices = actual_prices[1:]  # Remove the first element to match the length of predicted prices

forecasted_data = {'date':usa_data.index[window_size + 1:], 'actual':actual_prices, 'predicted':predicted_prices}
forecasted_data = pd.DataFrame(forecasted_data)

mae = mean_absolute_error(y_true = forecasted_data['actual'], y_pred = forecasted_data['predicted'])
print(f"Mean Absolute Error (MAE): {mae}")

mse = mean_squared_error(y_true = forecasted_data['actual'], y_pred = forecasted_data['predicted'])
print(f"Mean Squared Error (MSE): {mse}")

# plotting the forecasted price to the actual price with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(usa_data.index[window_size + 1:], actual_prices, label='Actual Price', alpha=0.6, color='black')
ax1.set_xlabel('Date')
ax1.set_ylabel('Actual Close Price', color='black')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
shifted_predicted_prices = [price * 0.98 for price in predicted_prices]  # Adjust the factor as needed
ax2.plot(usa_data.index[window_size + 1:], shifted_predicted_prices, label='Predicted Price', alpha=0.6, color='yellow', ls = '--')
ax2.set_ylabel('Predicted Close Price', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Actual vs Predicted Close Price')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.show()
