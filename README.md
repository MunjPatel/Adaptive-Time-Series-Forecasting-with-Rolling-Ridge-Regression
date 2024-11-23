# Adaptive-Time-Series-Forecasting-with-Rolling-Ridge-Regression

This repository contains the code accompanying the Medium article: [Adaptive Time Series Forecasting with Rolling Ridge Regression](https://medium.com/@patelmunj2011/adaptive-time-series-forecasting-with-rolling-ridge-regression-a82f4a718471).

## Overview

This repository contains the implementation of a Rolling Window Prediction Framework using Ridge Regression for financial time series forecasting. The framework dynamically adapts to evolving market conditions by leveraging a rolling window approach, where the model is trained on a moving window of recent data points. Key features of this repository include:

- Ridge Regression: A robust linear regression model with regularization, effective in capturing long-term trends while handling multicollinearity in financial datasets.
- Stochastic Residual Adjustments: Incorporates randomness into predictions to account for market volatility, making the forecasts more realistic and reflective of financial uncertainty.
- Dynamic Window Resizing: Implements iterative backward expansion of the training window to analyze the impact of increasing historical data on prediction performance.
- Performance Metrics: Includes the computation of Mean Squared Error (MSE) and Mean Absolute Error (MAE) across iterations, enabling a detailed evaluation of the model’s accuracy as the window size grows.
- Visualization: Provides clear visualizations of the model’s predictions versus actual prices, as well as the variation of MSE and MAE with training window size.

This repository is ideal for those looking to understand or implement dynamic regression-based forecasting methods in finance. The code is modular, easy to modify, and serves as a foundation for exploring extensions such as advanced feature engineering, hybrid models, or non-linear approaches.

## Repository Structure

- `requirements.txt`: List of all the dependencies for the project.
- `forecast.py`: Python script for financial time series forecasting using Ridge Regression.
- `plot_metrics.py`: Python script for plotting variation of MSE and MAE with training window size.

## Usage

The python scripts provide a comprehensive walkthrough of the analysis. Each script is self-contained and can be executed independently.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the readers of the accompanying Medium article for their feedback and support.

For a detailed explanation of the concepts and methodologies used, please refer to the original [article](https://medium.com/@patelmunj2011/adaptive-time-series-forecasting-with-rolling-ridge-regression-a82f4a718471)
