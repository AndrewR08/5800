import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    # fix random seed for reproducibility
    np.random.seed(8)
    tf.random.set_seed(8)

    df = pd.read_csv('data/__GAP1__.csv')
    df = df.tail(-1)
    df = df.drop(['Time', 'Distance_LEC', 'DistanceGap_SAI'], axis=1)

    X = df.values
    print(X.shape)

    lookback = 20

    Xs = []
    ys = []
    for i in range(lookback, len(df)):
        Xs.append(X[i-lookback:i])
        ys.append(X[i])
    X_train, y_train = np.array(Xs), np.array(ys)
    print(X_train.shape)

    """#walk forward model evaluation
    # for i in range(train_size, total_size):
    for i in range(train_size, train_size + 2):
        train, test = df.iloc[0:i], df.iloc[i:i + 1]
        print('train=%d, test=%d' % (len(train), len(test)))
        print(train)
        print(test)
        X_train, y_train = create_dataset(train, train.DistanceGap_SAI, time_steps)
        X_test, y_test = create_dataset(test, test.DistanceGap_SAI, time_steps)
        """

    patience = 20
    early_stopping = EarlyStopping(monitor='loss', patience=patience, verbose=1)

    keras.backend.clear_session()

    model = Sequential()
    model.add(LSTM(units=4, input_shape=(lookback, 1), activation='relu', return_sequences=True))
    model.add(LSTM(units=2, activation='relu'))
    model.add(Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer='adam')

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping])

    n_predictions = 10

    # Create the input data for prediction
    X_test = X[-lookback*2:-lookback].reshape(1, lookback, 1)
    print(X_test)

    # Make the predictions
    predictions = []
    for i in range(n_predictions):
        prediction = model.predict(X_test)[0][0]
        predictions.append(prediction)
        X_test = np.append(X_test[:, 1:, :], [[[prediction]]], axis=1)

    actual = X[-lookback:-lookback+n_predictions].flatten()

    # Print the predictions & actual
    print(predictions)
    print()
    print(actual)

    """plt.clf()
    plt.plot(np.arange(0, len(y_train)), y_train, 'g', label="history")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label="true")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.title('Distance - LSTM')
    plt.legend()
    plt.savefig('images/DistanceGapR1.png')"""

    #plt.clf()
    plt.plot(actual, label="true")
    plt.plot(predictions, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.title('Zoomed Distance SAI - LSTM')
    plt.legend()
    plt.savefig('images/Zoomed_DistanceL1.png')
    plt.show()


if __name__ == '__main__':
    main()
