import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

    return laps_pred


def fix_time(laps):
    for i in range(len(laps)):
        lap = laps.at[i, 'LapTime']
        if not pd.isna(lap):
            lap = lap[7:15]
            laps.at[i, 'LapTime'] = datetime.strptime(lap, '%H:%M:%S').second + \
                                    datetime.strptime(lap, '%H:%M:%S').minute * 60 \
                                    + datetime.strptime(lap, '%H:%M:%S').hour * 3600
        else:
            laps.at[i, 'LapTime'] = 0

    laps.drop(['Time', 'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
               'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL',
               'SpeedST', 'IsPersonalBest', 'Compound', 'TyreLife', 'FreshTyre', 'Stint', 'LapStartTime',
               'TrackStatus', 'IsAccurate', 'LapStartDate', 'PitLap'], axis=1, inplace=True)

    laps = laps[laps.LapTime != 0]
    return laps.reset_index()


def plot_lap_diff(laps, p_laps):
    x = laps['LapNumber'].values
    y1 = laps['LapTime'].values
    y2 = p_laps['LapTime'].values
    plt.scatter(x, y1, color='b', label='actual')
    plt.scatter(x, y2, color='r', alpha=0.5, label='predicted')
    plt.xlabel('Lap Number')
    plt.ylabel('Lap Time')
    plt.legend()
    plt.show()


def main():
    # True for pc / False for mac
    cache(False)

    laps = pd.read_csv('data/Monaco/Monaco_Grand_Prix.csv')
    laps = fix_time(laps)
    p_laps = predict_laps(laps)

    plot_lap_diff(laps, p_laps)


if __name__ == '__main__':
    main()
