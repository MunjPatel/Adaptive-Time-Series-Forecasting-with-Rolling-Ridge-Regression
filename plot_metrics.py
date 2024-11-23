import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from joblib import Parallel, delayed

ticker = yf.Ticker('^GSPC')
usa_data_full = ticker.history(period='max')

iterations = 30
start_year = 2014
results = {}

# function to compute metrics on rolling iterations
def compute_iteration(start_year, iteration, data):
    current_start_year = f"{start_year - iteration + 1}-01-01"
    end_year = "2024-01-01"

    usa_data = data[(data.index >= current_start_year) & (data.index < end_year)]

    window_size = 252
    alpha = 1.0

    usa_data['Lagged_Close'] = usa_data['Close'].shift(1)
    usa_data['Lagged_Volume'] = usa_data['Volume'].shift(1)
    usa_data = usa_data.dropna()

    X = usa_data[['Lagged_Close', 'Lagged_Volume']].values
    y = usa_data['Close'].values
    predicted_prices = []

    for i in range(window_size, len(X) - 1):
        X_train = X[i - window_size:i]
        y_train = y[i - window_size:i]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        residuals = y_train - model.predict(X_train)
        mu = residuals.mean()
        sigma = residuals.std()

        X_test = X[i + 1].reshape(1, -1)
        predicted_price = model.predict(X_test)[0] + mu + sigma * np.random.normal()
        predicted_prices.append(predicted_price)

    actual_prices = y[window_size + 1:]
    predicted_prices = np.array(predicted_prices[:len(actual_prices)])

    mae = mean_absolute_error(y_true=actual_prices, y_pred=predicted_prices)
    mse = mean_squared_error(y_true=actual_prices, y_pred=predicted_prices)
    return iteration, mse, mae

# parallelizing computations across multiple iterations
results_list = Parallel(n_jobs=-1)(
    delayed(compute_iteration)(start_year, iteration, usa_data_full) for iteration in range(1, iterations + 1)
)

# converting results into a dictionary
results = {iteration: [mse, mae] for iteration, mse, mae in results_list}

# converting results to a DataFrame for easier plotting
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MSE', 'MAE'])

# plotting MSE and MAE variation with iterations side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(results_df.index, results_df['MSE'], marker='o', label='MSE', color='blue')
axes[0].set_title('Variation of MSE with Iterations')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Mean Squared Error (MSE)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(results_df.index, results_df['MAE'], marker='o', label='MAE', color='red')
axes[1].set_title('Variation of MAE with Iterations')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Mean Absolute Error (MAE)')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
