import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import os
from os.path import exists

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def create_seq(X, lookback):
    Xs = []
    ys = []
    for i in range(lookback, lookback * 2):
        Xs.append(X[i - lookback:i])
        ys.append(X[i])
    return np.array(Xs), np.array(ys)


def main():
    train = False
    predict = True

    # fix random seed for reproducibility
    np.random.seed(8)
    tf.random.set_seed(8)

    df = pd.read_csv('data/__GAP2D__.csv')
    df = df.drop(['Time'], axis=1)

    #print(math.floor(len(df)/78))

    X = df.values
    print(X.shape)

    lookback = 150
    X_train, y_train = create_seq(X, lookback)
    print(X_train.shape)
    print(X_train)

    print("test start df: ", len(X_train)+2)

    features = len(X_train[0][0])

    if train:
        patience = 15
        early_stopping = EarlyStopping(monitor='loss', patience=patience, verbose=1)
        model_checkpoint = ModelCheckpoint('results/best_model2D.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)

        keras.backend.clear_session()

        model = Sequential()
        model.add(LSTM(units=64, input_shape=(lookback, features))) #, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Dense(units=features, activation='linear'))
        model.compile(
            loss='mean_squared_error',
            optimizer='rmsprop')

        model.fit(
            X_train, y_train,
            epochs=1000,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint])

    if predict:
        best_model = load_model('results/best_model2D.h5')

        print()

        n_predictions = 15

        test_start = len(X_train) + 2

        # Create the input data for prediction
        test_start = (len(X_train)*2) - 1
        X_test = X[test_start:test_start+lookback].reshape(1, lookback, features)
        print(X_test)
        print(X_test.shape)

        # Make the predictions
        predictions = []
        for i in range(n_predictions):
            prediction = best_model.predict(X_test)
            predictions.append(prediction.tolist())
            #print(X_test[:, 1:, :])
            #print(type(prediction))
            X_test = np.append(X_test[:, 1:, :], [prediction], axis=1)

        actual = X[test_start:test_start+n_predictions]

        # Flatten predictions list
        predictions = [element for sublist in predictions for element in sublist]

        # Print the predictions & actual
        print(predictions)
        print()
        print(actual)

        d1_true = []
        d2_true = []
        d1_pred = []
        d2_pred = []
        for i in range(len(predictions)):
            d1_true.append(actual[i][0])
            d2_true.append(actual[i][1])
            d1_pred.append(predictions[i][0])
            d2_pred.append(predictions[i][1])
        plt.plot(d1_true, label="d1 - true")
        plt.plot(d2_true, label="d2 - true")
        plt.plot(d1_pred, label="d1 - prediction")
        plt.plot(d2_pred, label="d2 - prediction")
        plt.ylabel('Distance Gap')
        plt.xlabel('Time Step')
        plt.title('Zoomed Distance Gap 2D - LSTM')
        plt.legend()
        plt.savefig('images/Zoomed_DistanceGap2D.png')
        plt.show()


if __name__ == '__main__':
    main()
