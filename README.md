# Sales Forecasting with Recurrent Neural Networks (RNN)

Sales forecasting is crucial for business planning and resource optimization. This repository explores the implementation of Long Short-Term Memory (LSTM) within the Recurrent Neural Networks (RNN) framework for time series prediction in sales forecasting. LSTM's ability to capture and remember long-term dependencies in sequential data is particularly relevant for forecasting tasks with inherent temporal patterns. In this project, I utilized 5 years of store item sales data to forecast sales.

## Overview

The following sections provide an overview of the project's key components and findings:

### Sales Distribution

![Sales Distribution](https://github.com/ManaswiniS/RNNBigData/assets/37972357/c27d3e5f-619e-4e13-89ec-797a4f9ee445)

The graph illustrates the distribution of sales across different stores and items over the dataset period.

### Hyperparameter Optimization

![Hyperparameter Sets Results](https://github.com/ManaswiniS/RNNBigData/assets/37972357/a2524c0e-dbab-4b7d-b2da-f205cf0bd933)

The table presents result comparisons across various hyperparameter sets. It highlights low Root Mean Square Error (RMSE) scores achieved with specific configurations, especially effective for less complex data. LSTM emerges as an optimal model choice for time series sales data.

### Sales Prediction Scatter Plot

![Scatter Plot](https://github.com/ManaswiniS/RNNBigData/assets/37972357/ca202b6f-a171-4c49-a67f-2a9f5da2598d)

This scatter plot predicts future sales values based on previous time series sales data, utilizing hyperparameters: Hidden State Size = 4, Epochs = 10, Learning rate = 0.1. The plot demonstrates the model's effectiveness in predicting values.

### Loss and RMSE over Epochs

![Loss and RMSE over Epochs](https://github.com/ManaswiniS/RNNBigData/assets/37972357/1412ab2b-35b8-425c-9a72-57b8dfc0f31b)

The plot displays Loss and RMSE values over epochs for the training data, employing hyperparameters: Hidden State Size = 4, Epochs = 10, Learning rate = 0.1.
