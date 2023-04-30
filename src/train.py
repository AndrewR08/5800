import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# fix random seed for reproducibility
np.random.seed(8)
tf.random.set_seed(8)


# function to format data for training lstm model using lookback parameter
# - X: input data
# - lookback: amount of data to use as history for model
# - returns: array of input data, array of output data
def create_seq(X, lookback):
    # create empty lists to append data
    Xs = []
    ys = []

    # append x, y data to lists from X based on lookback
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(X[i])
    return np.array(Xs), np.array(ys)


# function to train lstm model on given dataframe, input df & track infor
# - track info: track name, year, num_laps
def train(df, model_name):
    # drop unnecessary time column
    df = df.drop(['Time'], axis=1)

    # create array of df values
    X = df.values

    # define lookback (1 = 0.25s)
    lookback = 150  # ~40s of data / half lap

    # create training data using create_seq function
    X_train, y_train = create_seq(X, lookback)
    print(X_train.shape)
    #print(X_train)

    # get number of features for output size
    num_features = len(X_train[0][0])

    # define patience used for early stopping and initialize early stopping / best model saving
    patience = 20
    early_stopping = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    # *** use track name as model filename ***
    model_checkpoint = ModelCheckpoint('models/' + model_name + '.h5', monitor='loss', mode='min', verbose=1,
                                       save_best_only=True)

    # clear previous model training data to ensure best model outcomes
    keras.backend.clear_session()

    # define sequential model with 1 LSTM layer and Dense output layer
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(lookback, num_features), return_sequences=True))
    # add dropout to reduce overfitting
    model.add(Dropout(0.5))
    model.add(LSTM(units=32))
    model.add(Dense(units=num_features, activation='linear'))

    # compile model using mse as loss function and rmsprop as optimizer (better than adam for lstm)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # fit model using training data, can use high epochs w/ early stopping
    # - epochs: 1000 (early stopping will override this)
    # - batch_size: 32 (best fit from testing)
    # - verbose: 0 (1 to print output)
    # - callbacks: early stopping and model checkpoint saving
    model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint])

    # no need to return model, as best model will be saved as filename from model checkpoint
