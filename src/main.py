from train import *
from predict import *
from data import *
import fastf1 as ff1


def main():
    cache(True)     #True for desktop, False for mac
    year = 2022

    #get user input for track
    from data import lap_dict, track_dict
    keys = list(lap_dict.keys())
    for i in range(len(keys)):
        print((i+1), keys[i])
    inv_track_dict = {v: k for k, v in track_dict.items()}
    track_input = int(input("\n Select Race (1-" + str(len(keys)) + "): "))
    track = inv_track_dict[track_input]
    print(track)

    df_filename = 'GAP_2D.csv'

    circuits = get_schedule(year)

    ds = ['55', '4']
    #time_gap_race(year, track, drivers=ds, fn=df_filename)

    df = pd.read_csv('data/GAP_R.csv')
    model_name = 'test_model2D'
    train_size = 0.5
    verbose = 1         #0 for no output / 1 for output
    #train(df, model_name, train_size, verbose)

    model_path = 'results/' + model_name + '.h5'
    # predict(model_path, n_predictions, quali_df, lookback, n_features):

# **** finish predictions, calculate n_predictions(based on num laps/avg laptime)?,
# update menu interface w/ two columns for track name and input two drivers ****


if __name__ == '__main__':
    main()
