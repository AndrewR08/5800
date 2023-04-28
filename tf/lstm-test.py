# Import required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Prepare the data
# Suppose we have a time-series data 'X' with 100 time steps and one feature per time step
X = np.random.randn(100, 1)
print(X.shape)
print()

# Define the number of time steps to look back for making the predictions
n_lookback = 5

# Create the input data in the appropriate format for LSTM
X_train = []
y_train = []
for i in range(n_lookback, len(X)):
    X_train.append(X[i-n_lookback:i])
    y_train.append(X[i])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train)
print()
print(y_train)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Use the model for predictions
# Suppose we want to predict the next 10 time steps
n_predictions = 10

# Create the input data for prediction
X_test = X[-n_lookback:].reshape(1, n_lookback, 1)
print(X_test)

# Make the predictions
predictions = []
for i in range(n_predictions):
    prediction = model.predict(X_test)[0][0]
    #print(prediction)
    predictions.append(prediction)
    X_test = np.append(X_test[:, 1:, :], [[[prediction]]], axis=1)

# Print the predictions
print(predictions)

