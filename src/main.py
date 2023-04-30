from train import *
from predict import *
from data import *


def main():
    cache(True)     #True for desktop, False for mac
    year = 2022
    track = 'Monaco'

    ds = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18', '6', '24', '22', '23',
          '47', '20']
    #time_gap_race(year, track, drivers=ds)

    df = pd.read_csv('data/GAP_R.csv')
    model_name = 'test_modelR'
    train(df, model_name)

    model_path = 'results/' + model_name + '.h5'
    # predict(model_path, n_predictions, quali_df, lookback, n_features):


if __name__ == '__main__':
    main()
