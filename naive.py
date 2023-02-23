import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# options for easier readability on df print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option("expand_frame_repr", False)


def cache(pc):
    if pc:
        # location of cache for pc
        fastf1.Cache.enable_cache('D:/f1data')
    else:
        # location of cahce for mac
        fastf1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def predict_laps(df):
    laps_pred = pd.DataFrame(columns=df.columns)
    laps_pred['LapNumber'] = df['LapNumber']
    laps_pred['DriverNumber'] = df['DriverNumber']
    laps_pred['Driver'] = df['Driver']
    laps_pred['Team'] = df['Team']

    for i in range(len(laps_pred)):
        if i == 0:
            laps_pred.at[i, 'LapTime'] = df.at[i, 'LapTime']
        else:
            laps_pred.at[i, 'LapTime'] = df.at[i - 1, 'LapTime']

    """values = {'LapTime': '0 days 00:00:00.000000'}
    laps_pred = laps_pred.fillna(values)"""

    laps_pred.drop(['Time', 'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                    'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL',
                    'SpeedST', 'IsPersonalBest', 'Compound', 'TyreLife', 'FreshTyre', 'Stint', 'LapStartTime',
                    'TrackStatus', 'IsAccurate', 'LapStartDate', 'PitLap'], axis=1, inplace=True)
    return laps_pred


def plot_lap_diff(laps, p_laps):
    fig, axes = plt.subplots(1, 2)
    #print(laps['LapTime'].to_numpy())
    x = laps['LapNumber'].values
    y = laps['LapTime'].values
    print(x.shape)
    axes[0, 0].plot(x, y)
    axes[0, 0].set_title('Actual Laps')
    axes[0, 0].set_xlabel('Lap Number')
    axes[0, 0].set_ylabel('Lap Time')

    axes[0, 1].plot(p_laps['LapNumber'].values, p_laps['LapTime'].values)
    axes[0, 1].set_title('Predicted Laps')
    axes[0, 1].set_xlabel('Lap Number')
    axes[0, 1].set_ylabel('Lap Time')
    plt.tight_layout()


def main():
    # True for pc / False for mac
    cache(False)

    laps = pd.read_csv('data/Monaco/Monaco_Grand_Prix.csv')
    p_laps = predict_laps(laps)
    #print(p_laps)

    plot_lap_diff(laps, p_laps)


if __name__ == '__main__':
    main()
