# Import required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate some toy data
n_samples = 100
n_lookback = 5
n_features = 2
X = np.random.randn(n_samples, n_lookback, n_features)
print(X.shape)
y1 = np.random.randn(n_samples, 1)
y2 = np.random.randn(n_samples, 1)
y3 = np.random.randn(n_samples, 1)
y = np.hstack((y1, y2, y3))

# Split the data into training and test sets
n_train = int(0.8 * n_samples)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Build the model
model = Sequential()
model.add(LSTM(32, input_shape=(n_lookback, n_features)))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Use the model for predictions
# Suppose we want to predict the next 10 time steps
n_predictions = 10

# Create the input data for prediction
X_test_input = X[-n_lookback:].reshape(1, n_lookback, n_features)

# Make the predictions
predictions = []
for i in range(n_predictions):
    prediction = model.predict(X_test_input)[0]
    predictions.append(prediction)
    X_test_input = np.append(X_test_input[:, 1:, :], [[prediction]], axis=1)

# Print the predictions
print(predictions)

