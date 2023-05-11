# Options Price Forecasting

## Description
The purpose of the models in this project is to predict the prices of TSLA options. A total of 4 models were trained to predict call prices:
1. Multilayer Perceptron
2. Simple Recurrent Neural Network (SimpleRNN)
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)
 
## Repository Structure
The repository is organized as such:
```
├── models
│	├── mlp_calls
│	├── mlp_puts
│	├── rnn_calls
│	└── 
└── notebooks
	├── mlp.ipynb
	├── preprocessing.ipynb
	└── rnn.ipynb
```

The models that were built are stored in the `models` folder. There are 4 models:
1. `mlp_calls`: multilayer perceptron model for predicting call prices.
2. `mlp_puts`: multilayer perceptron model for predicting put prices.
3. `rnn_calls`: recurrent neural network model for predicting call prices.

The notebooks containing our code and implementation are stored in the `notebooks` folder. There are 3 notebooks:
1. `mlp.ipynb`: code for MLP models (includes initial model, hyperparameter tuning, and final model)
2. `preprocessing.ipynb`: code for cleaning and preprocessing original dataset (includes feature engineering) 
3. `rnn.ipynb`: code for RNN models (includes Simple RNN, LSTM, and GRU)

The datasets we used can be found here: https://drive.google.com/drive/folders/14UDaBowufW9BTIz6CvdKxtkL2l0c_DAB?usp=sharing

There are 5 CSV files:
 1. `options_tsla.csv`: the original dataset on TSLA options. This dataset was cleaned and preprocessed to create the other datasets in the folder.
 2. `calls_data.csv`: feature data for TSLA calls.
 3. `calls_target.csv`: target data for TSLA calls.
 4. `puts_data.csv`: feature data for TSLA puts.
 5. `puts_target.csv`: target data for TSLA puts.

## Executing the Code
To retrain the models, first download the data. The data then needs to be prepared for the various models. This can be done by running each cell in notebooks/preprocessing.ipynb in order.

To train the multilayer perceptron model, run each cell in notebooks/mlp.ipynb. Similarly, to train the SimpleRNN, LSTM, or GRU model, run the corresponding cell in notebooks/rnn.ipynb.

Alternatively, to load our pretrained models (stored in the `models` folder), run the following code:
```
import tensorflow as tf

model = tf.keras.models.load_model('path/to/model')
```

## Results and Observations
Results (including charts/tables) and your observations