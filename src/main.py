from train import *
from predict import *
from data import *
import fastf1 as ff1


def main():
    cache(True)     #True for desktop, False for mac
    year = 2022

    # Get user input for track using track dictionary
    from data import race_dict
    keys = list(race_dict.keys())
    for i in range(len(keys)):
        print((i+1), keys[i])

    inv_race_dict = {v: k for k, v in race_dict.items()}
    race_input = int(input("\n Select Race (1-" + str(len(keys)) + "): "))
    track = inv_race_dict[race_input]
    print(track)

    # Set filename to save dataframe as csv
    df_filename = 'GAP_2D.csv'

    # Get user input for drivers
    ds = ['55', '4']

    # Use data.py function to create distance gap csv
    #time_gap_race(year, track, drivers=ds, fn=df_filename)

    # Read distance gap csv
    df = pd.read_csv('data/GAP_R.csv')

    # Set model parameters for training model
    model_name = 'test_model2D'
    train_size = 0.5
    verbose = 1         #0 for no output / 1 for output
    #train(df, model_name, train_size, verbose)

    # Set model parameters for predicting with model
    model_path = 'results/' + model_name + '.h5'
    # predict(model_path, n_predictions, quali_df, lookback, n_features):

# **** finish predictions, calculate n_predictions(based on num laps/avg laptime)?,
# update menu interface w/ two columns for track name and input two drivers ****


if __name__ == '__main__':
    main()
