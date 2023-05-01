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
# - df: data to be used as history for model predictions
# - lookback: amount of data to use as history for model
# - n_features: number of features for output size
# - train_size: amount of data withheld for training, use rest for testing
def predict(model_path, n_predictions, df, lookback, X_train):
    model = load(model_path)

    df = df.drop(['Time'], axis=1)
    X = df.values
    n_features = len(X_train[0][0])
    test_start = (len(X_train) * 2) - 1
    X_test = X[test_start:test_start+lookback].reshape(1, lookback, n_features)

    predictions = []
    for i in range(n_predictions):
        prediction = model.predict(X_test)
        predictions.append(prediction.tolist())
        X_test = np.append(X_test[:, 1:, :], [prediction], axis=1)

    # Flatten predictions list
    predictions = [element for sublist in predictions for element in sublist]

    print(test_start, n_predictions)
    actual = X[test_start:test_start+n_predictions]

    return predictions, actual


# function to plot prediction values
# - predictions: array of predicted values from predict() function
# - actual: array of actual values in data from predict() function
# - filename: filename to save plot image
# - ds: list of driver identifiers
def plot_pred(predictions, actual, filename, ds):
    d1_true = []
    d1_pred = []
    for i in range(len(predictions)):
        #d1_true.append(actual[i])
        d1_pred.append(predictions[i])
    #plt.plot(d1_true, label=ds[0] + " - true")
    plt.plot(d1_pred, label=ds[0] + " - prediction")
    plt.ylabel('Distance Gap')
    plt.xlabel('Time Step')
    plt.title('Zoomed Distance Gap Prediction ' + ds[0] + "/" + ds[1])
    plt.legend()
    #plt.savefig('images/'+filename)
    plt.show()


def evaluate_model(model_path):
    model = load_model(model_path)
    print(model.evaluate())
