import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import load_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# function to load model from training if it exists, otherwise print ERROR message
# - file_path: path to model.h5 file
def load(file_path):
    if os.path.exists(file_path):
        best_model = load_model(file_path)
    else:
        print("ERROR: File Not Found")
    return best_model


# function to predict using loaded model and data from quali lap
# - model_path: path to model in which to load
# - n_predictions: number of predictions to make (1 = 0.25s)
# - quali_df: qualifying data to be used as history for model predictions
# - lookback: amount of data to use as history for model
# - n_features: number of features for output size
def predict(model_path, n_predictions, quali_df, lookback, n_features):
    model = load(model_path)

    X = quali_df.values
    X_test = X.reshape(1, lookback, n_features)

    predictions = []
    for i in range(n_predictions):
        prediction = model.predict(X_test)
        predictions.append(prediction.tolist())
        X_test = np.append(X_test[:, 1:, :], [prediction], axis=1)

    # Flatten predictions list
    predictions = [element for sublist in predictions for element in sublist]


# function to plot prediction values
# - predictions: array of predicted values from predict() function
# - filename: filename to save plot image
def plot_pred(predictions, filename):
    d1_pred = []
    d2_pred = []
    for i in range(len(predictions)):
        d1_pred.append(predictions[i][0])
        d2_pred.append(predictions[i][1])
    plt.plot(d1_pred, label="d1 - prediction")
    plt.plot(d2_pred, label="d2 - prediction")
    plt.ylabel('Distance Gap')
    plt.xlabel('Time Step')
    plt.title('Zoomed Distance Gap Prediction - LSTM')
    plt.legend()
    plt.savefig('images/'+filename)
    plt.show()


def evaluate_model(model_path):
    model = load_model(model_path)
    print(model.evaluate())