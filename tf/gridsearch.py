#!/usr/bin/python
import shutil
import itertools
import tensorflow
import keras
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.losses import *
from random import shuffle
from os.path import exists
import numpy as np
import pandas as pd
from helper import *

# fix random seed for reproducibility
np.random.seed(8)
tf.random.set_seed(8)

# Constants - (ACCEPTABLE ERROR)
optimizer = 'adam'
batch_size = 32
epochs = 200

# Debug settings
PRINT_PERMUTATIONS = True  # Whether to print the amount of permutations while running
RANDOM_ORDERING = True  # Whether to grid search in random order (good for faster discovery)


def run(train, test, layers, loss_function, optimizer, batch_size, epochs, save, patience=15):
    # Clear backend
    keras.backend.clear_session()

    # Printing Debug information
    num_gpus = len(tensorflow.config.experimental.list_physical_devices('GPU'))
    using_gpus = num_gpus >= 1
    print(f"Using GPU: {using_gpus}\n")

    # print(layers[0]._keras_api_names[0][13:])
    # Debug variables
    layers_str = "[" + "|".join(str(str(x.units) + " " + x._keras_api_names[0][13:]) for x in layers) + "]"
    loss_function_name = loss_function.name
    print(f"hyper-parameters:\n\t" +
          f"Layers: {layers_str}\n\tLoss Function: {loss_function_name}\n\tBatch Size: {batch_size}\n\t" +
          f"Epochs: {epochs}")

    # Setup path for artifacts
    output_path = f'checkpoints/artifacts'
    # Prune previous attempts
    if exists(output_path):
        shutil.rmtree(output_path)

    # Get x, y
    time_steps = 10

    X_train, y_train = create_dataset(train, train.DistanceGap_SAI, time_steps)
    X_test, y_test = create_dataset(test, test.DistanceGap_SAI, time_steps)

    # Setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    # Sequential Model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2]), name='input'))

    # Hidden Layers
    for layer in layers:
        model.add(layer)

    # Add output layer
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.optimizer = optimizer
    model.compile(loss=loss_function)
    model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[model_checkpoint, early_stopping])
    number_of_epochs_ran = len(history.history['val_loss'])
    val_loss = model.evaluate(X_test, y_test, verbose=0)

    # Write result to results
    csv_result = f"{layers_str},{loss_function_name},{batch_size},{epochs},{number_of_epochs_ran},{val_loss}\n"
    file1 = open('results_race.csv', 'a+')
    file1.write(csv_result)
    file1.close()
    print("Results appended.\n")

    # At the end, get the best model and visualize it.
    model, best_epoch, best_loss = get_best_model(output_path)
    print(f"The best model was discovered on epoch {best_epoch} and had a loss of {best_loss}")

    # Check if the current model is better or not
    prev_best_val_loss = float('inf')
    if exists(f"best_models/gap1all.h5"):
        prev_best_model = keras.models.load_model(f"best_models/gap1race.h5", compile=True,
                                                  custom_objects={'Normalization': Normalization})
        prev_best_val_loss = prev_best_model.evaluate(X_test, y_test, verbose=0)
    if prev_best_val_loss - best_loss > 0.000001:
        print(f"NEW RECORD! Loss: {best_loss}, saved to: best_models/gap1race.h5")
        model.save(f"best_models/gap1race.h5")
    else:
        print(f"This run did not beat the previous best loss of {prev_best_val_loss}")

    # Save result
    if save:
        print("Saving visualizations of the best model...")
        # Save visualizations of the best model
        visualize(model, x=X_train, y_true=y_train, name='Training', output_path=output_path)
        print(f"Saved " + str(os.path.join(output_path, f'visualize_Training.png')) + '.')

        training_png_file_path = os.path.join(output_path, f'visualize_Validation.png')
        visualize(model, x=X_test, y_true=y_test, name='Validation', output_path=output_path)
        print(f"Saved " + str(os.path.join(output_path, f'visualize_Validation.png')) + '.')

    return best_loss


# Run grid search
def grid_search(train, test, layer_counts, neuron_counts, loss_functions):
    if PRINT_PERMUTATIONS:
        amt_loss_functions = len(loss_functions)
        amt_neuron_counts = len(neuron_counts)
        amt_total = 0
        for layer_count in layer_counts:
            amt_neuron_total = amt_neuron_counts ** layer_count
            amt_total += (amt_neuron_total)
        amt_total *= amt_loss_functions
        print(f"Total permutations: {amt_total}")

    layer_permutations = []
    print("Calcuating permutations...")
    for loss_function in loss_functions:
        for layer_count in layer_counts:
            neuron_count_permutations = list(itertools.product(neuron_counts, repeat=layer_count))

            perms = list(neuron_count_permutations)
            for layer_neuron_counts in perms:
                layers = []
                for i in range(len(layer_neuron_counts)):
                    neuron_amt = layer_neuron_counts[i]
                    layer_name = "layer" + str(len(layers))
                    layers.append(LSTM(units=neuron_amt, return_sequences=True, name=layer_name))
                layer_permutations.append(layers)
    amt_layer_permutations = len(layer_permutations)
    print(f"All {amt_layer_permutations} permutations compiled.")

    if RANDOM_ORDERING:
        print("Randomizing permutation order...")
        shuffle(layer_permutations)
        print("Randomized.")
    print("Beginning grid search...")
    for layer_permutation in layer_permutations:
        run(train, test, layer_permutation, loss_function, optimizer, batch_size, epochs, False)
    print("Grid search complete.")


def main():
    df = pd.read_csv('data/__GAP1R__.csv')
    df = df.tail(-1)
    df = df.drop(['Distance_LEC', 'Distance_SAI'], axis=1)      #not needed for gap1r

    train_size = int(len(df) * 0.5)
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    # Hyperparameters for Grid Search
    layer_counts = [1, 2, 3, 4]
    neuron_counts = [128, 256, 512]
    loss_functions = [MeanSquaredError()]

    grid_search(train, test, layer_counts, neuron_counts, loss_functions)


if __name__ == '__main__':
    main()
