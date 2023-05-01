import shutil
from train import *
from predict import *
from data import *
import cmd
import timeit


def main():
    cache(True)  # True for desktop, False for mac

    # set year for race data
    year = 2022

    # Display race options menu
    r_keys = list(race_dict.keys())
    for i in range(len(r_keys)):
        r_keys[i] = str(i + 1) + " - " + r_keys[i]
    cli = cmd.Cmd()
    cli.columnize(r_keys, displaywidth=80)

    # Get user input for track using track dictionary
    inv_race_dict = {v: k for k, v in race_dict.items()}
    race_input = int(input("\nSelect Race (1-" + str(len(r_keys)) + "): "))
    track = inv_race_dict[race_input]

    # Display driver options menu
    print()
    ds = []
    d_keys = list(drivers_dict.keys())
    d_vals = list(drivers_dict.values())
    for i in range(len(d_keys)):
        d_vals[i] = str(d_keys[i]) + " - " + d_vals[i]
    cli = cmd.Cmd()
    cli.columnize(d_vals, displaywidth=20)

    # Get user input for drivers, ensuring valid response
    again1 = True
    again2 = True
    while again1 or again2:
        if again1:
            d1_input = input("\nSelect Lead Driver Number: ")
            if int(d1_input) not in d_keys:
                print("Please Select a Valid Driver Number")
            else:
                again1 = False
        elif again2:
            d2_input = input("Select Second Driver Number: ")
            if d2_input == d1_input:
                print("Cannot Select the Same Driver Number")
            elif int(d2_input) not in d_keys:
                print("Please Select a Valid Driver Number")
            else:
                again2 = False
        else:
            again1 = False
            again2 = False
    ds.append(d1_input)
    ds.append(d2_input)


    # Set filename to save dataframe as csv
    df_filename = 'GAP_2D.csv'

    #start timing
    starttime = timeit.default_timer()

    # Use data.py function to create distance gap csv
    time_gap_race(year=year, track=track, drivers=ds, fn=df_filename, num_laps=1)

    # Read created distance gap csv
    df = pd.read_csv('data/' + df_filename)

    # Set model parameters for training model
    model_name = 'test_model2D'
    train_size = 0.5
    lookback = 20   # 20=~5s of data
    verbose = 1     # 0 for no output / 1 for output
    X_train = train(df, model_name, train_size, lookback, verbose)

    # Set model parameters for predicting with model
    model_path = 'models/' + model_name + '.h5'
    n_predictions = 15      # 15 ~= 3.75s of data (300 ~= 1 lap)
    n_predictions = int(train_size*len(X_train))
    print(n_predictions)

    # Call predict() function and store prediction and actual values
    predictions, actual = predict(model_path, n_predictions, df, lookback, X_train)

    # print time difference
    print("The time difference is :", timeit.default_timer() - starttime)

    # Display results between predictions and actual
    filename = 'test_img.png'
    plot_pred(predictions, actual, filename, ds)

# ****  - update train csv to be entire race but only use first laps to train and rest to compare predictions with
#       - save dataframes, models, etc based on input (ex. British GP, SAI/LEC = 'British_Grand_Prix_55_16'


if __name__ == '__main__':
    main()
