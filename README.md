# Options Price Forecasting

## Description
The purpose of the models in this project is to predict option prices. 
 
## Repository Structure
The repository is organized as such:
```
├── data
│   ├── calls_data.csv
│   ├── calls_target.csv
│   ├── options_tsla.csv
│   ├── puts_data.csv
│   └── puts_target.csv
├── models
│	├── mlp_calls
│	├── mlp_puts
│	├── rnn_calls
│	├── rnn_puts
│   └── 
└── notebooks
	├── mlp.ipynb
	├── preprocessing.ipynb
	└── rnn.ipynb
```
 The datasets we used for this project are saved as CSVs, and are stored in the `data` folder. There are 5 CSV files:
 1. `options_tsla.csv`: the original dataset on TSLA options. This dataset was cleaned and preprocessed to create the other datasets in the folder.
 2. `calls_data.csv`: feature data for TSLA calls.
 3. `calls_target.csv`: target data for TSLA calls.
 4. `puts_data.csv`: feature data for TSLA puts.
 5. `puts_target.csv`: target data for TSLA puts.

The models that were built are stored in the `models` folder. There are 4 models:
1. `mlp_calls`: multilayer perceptron model for predicting call prices.
2. `mlp_puts`: multilayer perceptron model for predicting put prices.
3. `rnn_calls`: recurrent neural network model for predicting call prices.
4. `rnn_puts`: recurrent neural network model for predicting call prices.

The notebooks containing our code and implementation are stored in the `notebooks` folder. There are 3 notebooks:
1. `mlp.ipynb`: code for MLP models (includes initial model, hyperparameter tuning, and final model)
2. `preprocessing.ipynb`: code for cleaning and preprocessing original dataset (includes feature engineering) 
3. `rnn.ipynb`: code for RNN models

## Executing the Code
Example commands to execute the code
 
## Results and Observations
Results (including charts/tables) and your observations