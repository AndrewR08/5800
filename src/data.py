import fastf1 as ff1
import pandas as pd
import numpy as np
import datetime
import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

lap_dict = {'Abu_Dhabi_Grand_Prix': 5554, 'Australian_Grand_Prix': 5303,
            'Austrian_Grand_Prix': 4318, 'Azerbaijan': 6003, 'Bahrain_Grand_Prix': 5412, 'Belgian_Grand_Prix': 7004,
            'British_Grand_Prix': 5891, 'Canadian_Grand_Prix': 4361,
            'Dutch_Grand_Prix': 4259, 'Emilia_Romagna_Grand_Prix': 4909, 'French_Grand_Prix': 5842,
            'Hungarian_Grand_Prix': 4381, 'Italian_Grand_Prix': 5793, 'Japanese_Grand_Prix': 5807,
            'Mexico_City_Grand_Prix': 4304, 'Miami_Grand_Prix': 5410, 'Monaco_Grand_Prix': 3337,
            'Sao_Paulo_Grand_Prix': 4309, 'Saudi_Arabian_Grand_Prix': 6175, 'Singapore_Grand_Prix': 5063,
            'Spanish_Grand_Prix': 4655, 'United_States_Grand_Prix': 5513}

race_dict = {'Abu_Dhabi_Grand_Prix': 1, 'Australian_Grand_Prix': 2,
             'Austrian_Grand_Prix': 3, 'Azerbaijan_Grand_Prix': 4, 'Bahrain_Grand_Prix': 5, 'Belgian_Grand_Prix': 6,
             'British_Grand_Prix': 7, 'Canadian_Grand_Prix': 8,
             'Dutch_Grand_Prix': 9, 'Emilia_Romagna_Grand_Prix': 10, 'French_Grand_Prix': 11,
             'Hungarian_Grand_Prix': 12, 'Italian_Grand_Prix': 13, 'Japanese_Grand_Prix': 14,
             'Mexico_City_Grand_Prix': 15, 'Miami_Grand_Prix': 16, 'Monaco_Grand_Prix': 17,
             'Sao_Paulo_Grand_Prix': 18, 'Saudi_Arabian_Grand_Prix': 19, 'Singapore_Grand_Prix': 20,
             'Spanish_Grand_Prix': 21, 'United_States_Grand_Prix': 22}

drivers_dict = {1: 'VER', 3: 'RIC', 4: 'NOR', 5: 'VET', 6: 'LAT', 10: 'GAS', 11: 'PER', 14: 'ALO', 16: 'LEC', 18: 'STR',
                20: 'MAG', 22: 'TSU', 23: 'ALB', 24: 'ZHO', 31: 'OCO', 44: 'HAM', 47: 'MSC', 55: 'SAI', 63: 'RUS',
                77: 'BOT'}


# function to determine cache location of fastf1 data
# - pc: True = Desktop, False = Mac
def cache(pc):
    if pc:
        # location of cache for pc
        ff1.Cache.enable_cache('D:/f1data')
    else:
        # location of cache for mac
        ff1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def format_circuits(circuit):
    if isinstance(circuit, str):
        circuit = circuit.replace(' ', '_')
    return circuit


def get_schedule(year):
    schedule = ff1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']
    circuits = schedule.EventName
    circuits = circuits.apply(format_circuits).reset_index(drop=True)
    circuits = circuits.to_dict()
    return circuits


def remove_outliers(df, columns, n_std):
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean + (n_std * sd))]
    return df


def time_gap_race(year, track, drivers, fn, num_laps=None):
    # cache(True)

    # race distance variable
    race_dist = lap_dict[track]
    # print(race_dist)

    laps_df = pd.DataFrame()

    session = ff1.get_session(year, track, 'R')
    session.load()

    for i in range(len(drivers)):
        d = drivers[i]
        d_laps = session.laps.pick_driver(d)
        if num_laps is None:
            laps = int(max(d_laps.LapNumber))
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
            # laps_df.to_csv('data/MonacoTD_lapsdf.csv', index=False)

    max_time = math.ceil(max(laps_df['Time']).total_seconds())
    # print("max time, ", max_time)
    laps_df_new = pd.DataFrame()
    t = np.linspace(0, max_time, math.floor(max_time / 0.25))  # use 0.25s sample rate
    laps_df_new['Time'] = t

    for d in drivers:
        tp = laps_df['Time'].loc[laps_df['DriverNumber'] == d].apply(lambda row: row.total_seconds())
        dp = laps_df['Distance'].loc[laps_df['DriverNumber'] == d]
        d_new = np.interp(t, tp, dp)
        laps_df_new['Distance_' + str(laps_df.Driver.loc[laps_df.DriverNumber == d].iloc[0])] = d_new

    df = laps_df_new
    # df.to_csv('data/MonacoTD_df.csv', index=False)

    # df['Time'] = df['Time'].apply(lambda row: datetime.timedelta(seconds=row))
    leader = str(df.columns[1])[9:]
    # print("leader, ", leader)
    for i in range(len(drivers)):
        d = str(df.columns[i + 1])[9:]
        # print(d)
        if d != leader:
            df['DistanceGap_' + d] = df['Distance_' + leader] - df['Distance_' + d]
            df = remove_outliers(df, ['DistanceGap_' + d], n_std=4)

    df.drop(columns=df.columns[1:(len(drivers) + 1)], inplace=True)

    df.to_csv('data/' + fn, index=False)
