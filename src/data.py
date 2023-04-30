import fastf1 as ff1
import pandas as pd
import numpy as np
import datetime
import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


# function to determine cache location of fastf1 data
# - pc: True = Desktop, False = Mac
def cache(pc):
    if pc:
        # location of cache for pc
        ff1.Cache.enable_cache('D:/f1data')
    else:
        # location of cache for mac
        ff1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def remove_outliers(df, columns, n_std):
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean + (n_std * sd))]
    return df


def time_gap_race(year, track, drivers, num_laps=None):
    #cache(True)

    # race distance variable
    race_dist = 3337

    laps_df = pd.DataFrame()

    session = ff1.get_session(year, track, 'R')
    session.load()

    for i in range(len(drivers)):
        d = drivers[i]
        d_laps = session.laps.pick_driver(d)
        if num_laps is None:
            laps = max(d_laps.LapNumber)
        else:
            laps = num_laps

        # use range(1, laps+1) to exclude lap 1
        for j in range(1, laps + 1):
            if j < laps:
                d_lap = d_laps[d_laps['LapNumber'] == (j + 1)].iloc[0]
            else:
                d_lap = d_laps[d_laps['LapNumber'] == j].iloc[0]

            ref = d_lap.get_car_data(interpolate_edges=True)
            ref = ref.add_distance()
            ref = ref[['Time', 'Distance']]

            # use j != 1 to exclude lap 1
            if j != 1:
                ref_len = len(laps_df)
                last_ref_dist = laps_df['Distance'].iloc[ref_len - 1]
                last_ref_time = laps_df['Time'].iloc[ref_len - 1]
                ref['Time'] = ref['Time'] + last_ref_time
                ref['Distance'] = ref['Distance'] + last_ref_dist

            ref['LapNumber'] = d_lap.LapNumber
            ref['Driver'] = d_laps.Driver.iloc[0]
            ref['DriverNumber'] = d_laps.DriverNumber.iloc[0]
            max_dist = max(ref['Distance'])
            ref_dist = race_dist * d_lap.LapNumber
            ref['Distance'] = ref['Distance'].apply(lambda row: (ref_dist / max_dist) * row)

            laps_df = pd.concat([laps_df, ref]).reset_index(drop=True)

    laps_df['TimeDiff'] = laps_df['Time'].diff()
    laps_df['TimeDiff'] = laps_df['TimeDiff'].apply(lambda row: row.total_seconds())
    laps_df['TimeDiff'].fillna(0)
    laps_df['TimeDiff'].loc[laps_df['TimeDiff'] < 0] = 0

    # print(laps_df)
    # laps_df.to_csv('data/Monaco/MonacoTD_lapsdf_TEST3.csv', index=False)

    max_time = math.ceil(max(laps_df['Time']).total_seconds())
    # print("max time, ", max_time)
    laps_df_new = pd.DataFrame()
    t = np.linspace(0, max_time, math.floor(max_time / 0.25))  # use 0.18s sample rate
    laps_df_new['Time'] = t

    for d in drivers:
        tp = laps_df['Time'].loc[laps_df['DriverNumber'] == d].apply(lambda row: row.total_seconds())
        dp = laps_df['Distance'].loc[laps_df['DriverNumber'] == d]
        d_new = np.interp(t, tp, dp)
        laps_df_new['Distance_' + str(laps_df.Driver.loc[laps_df.DriverNumber == d].iloc[0])] = d_new

    df = laps_df_new
    # df.to_csv('data/Monaco/MonacoTD_lapsdf_TEST3.csv', index=False)

    # df['Time'] = df['Time'].apply(lambda row: datetime.timedelta(seconds=row))
    leader = str(df.columns[1])[9:]
    # print("leader, ", leader)
    for i in range(len(drivers)):
        d = str(df.columns[i + 1])[9:]
        # print(d)
        if d != leader:
            df['DistanceGap_' + d] = df['Distance_' + leader] - df['Distance_' + d]
            # df['DistanceGap_'+d] = df['DistanceGap_'+d].loc[df['DistanceGap_'+d] > 3337].apply(lambda row: 0)
            # df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
            df = remove_outliers(df, ['DistanceGap_' + d], n_std=4)

    df.drop(columns=df.columns[1:(len(drivers) + 1)], inplace=True)

    df.to_csv('data/GAP_R.csv', index=False)


def time_gap_quali(year, track, drivers):
    # cache(True)

    # race distance variable
    race_dist = 3337

    laps_df = pd.DataFrame()

    session = ff1.get_session(year, track, 'R')
    session.load()
