import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        vals = X.iloc[i:(i + time_steps)].values
        Xs.append(vals)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    # fix random seed for reproducibility
    np.random.seed(8)
    tf.random.set_seed(8)

    df = pd.read_csv('data/__GAP1__.csv')
    df = df.tail(-1)
    df = df.drop(['Distance_LEC', 'Distance_SAI'], axis=1)

    """best_model = keras.models.load_model(f"best_models/gap1all.h5", compile=True,
                                              custom_objects={'Normalization': Normalization})
    print(best_model)"""

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train)

    time_steps = 10
    X_train, y_train = create_dataset(train, train.DistanceGap_SAI, time_steps)
    X_test, y_test = create_dataset(test, test.DistanceGap_SAI, time_steps)

    keras.backend.clear_session()

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2]), name='input'))
    model.add(LSTM(units=512, return_sequences=True))     #, activation='relu'))
    model.add(LSTM(units=512, return_sequences=True))     #, activation='relu'))
    model.add(LSTM(units=256))                              #, activation='relu'))

    #model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(units=1))
    model.compile(
      loss='mean_squared_error',
      optimizer='rmsprop')

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False)

    """plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('images/Epoch_Loss.png')"""

    y_pred = model.predict(X_test, verbose=1)
    print(y_pred[0])
    print(y_train[0])


    plt.clf()
    plt.plot(np.arange(0, len(y_train)), y_train, 'g', label="history")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label="true")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.title('Distance - LSTM')
    plt.legend()
    plt.savefig('images/DistanceGapR1.png')

    plt.clf()
    plt.plot(y_test, label="true")
    plt.plot(y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.title('Zoomed Distance SAI - LSTM')
    plt.legend()
    plt.savefig('images/Zoomed_DistanceGapR1.png')


if __name__ == '__main__':
    main()