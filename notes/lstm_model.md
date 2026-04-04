# LSTM

## Neural Networks

A neural network is a machine learning model composed of layers of interconnected units (neurons). Each neuron applies a weighted sum of its inputs followed by a non-linear activation function. Networks are trained by backpropagation: computing the gradient of the loss with respect to all weights and updating them iteratively.

## Recurrent Neural Networks (RNN)

A standard neural network processes each input independently. A Recurrent Neural Network (RNN) introduces a hidden state that is passed from one time step to the next, allowing the network to retain information about previous inputs. This makes RNNs naturally suited for sequential data like time series.

However, standard RNNs struggle to learn long-range dependencies due to the **vanishing gradient problem**: gradients shrink exponentially as they are propagated back through many time steps, making it hard for the network to learn from distant past observations.

## LSTM

LSTM (Long Short-Term Memory) is a type of RNN designed to address the vanishing gradient problem. It introduces a **cell state** — a separate memory that runs through all time steps — and three **gates** that control how information flows:

- **Forget gate:** decides what information to discard from the cell state.
- **Input gate:** decides what new information to store in the cell state.
- **Output gate:** decides what part of the cell state to output as the hidden state.

This gating mechanism allows the LSTM to retain relevant information over long sequences and forget irrelevant information.

### LSTM for Regression

For time series regression, the LSTM receives a sequence of past observations (a sliding window) and outputs a single predicted value. The output layer is a single dense neuron with no activation function (linear output).

### Some configurable Hyperparameters

| Hyperparameter | Description |
|---|---|
| `units` | Number of memory units (neurons) in the LSTM layer; controls model capacity |
| `window_size` | Number of past time steps fed as input |
| `epochs` | Number of full passes through the training data |
| `batch_size` | Number of samples per gradient update |
| `learning_rate` | Step size for the optimizer (typically Adam) |
| `dropout` | Fraction of units randomly set to zero during training to reduce overfitting |