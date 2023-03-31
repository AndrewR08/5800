import fastf1 as ff1
import pandas as pd
from fastf1 import plotting
from fastf1 import utils
from matplotlib import pyplot as plt
import numpy as np
import datetime
import math

# options for easier readability on df print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option("expand_frame_repr", False)


def cache(pc):
    if pc:
        # location of cache for pc
        ff1.Cache.enable_cache('D:/f1data')
    else:
        # location of cahce for mac
        ff1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def driver_ahead_test(year, race, drivers):
    session = ff1.get_session(year, race, 'R')
    session.load()

    tel_df = pd.DataFrame()
    d_laps = session.laps.pick_driver(drivers[0])
    lap_num = 1
    d_lap = d_laps[d_laps['LapNumber'] == lap_num].iloc[0]
    ref = d_lap.get_telemetry()
    ref.to_csv('data/test_ref.csv', index=False)

    """for i in range(len(drivers)):
        d = drivers[i]
        d_laps = session.laps.pick_driver(d)

        laps = max(d_laps.LapNumber)
        for j in range(laps):
            d_lap = d_laps[d_laps['LapNumber'] == j+1].iloc[0]

            ref = d_lap.get_telemetry()
            ref['LapNumber'] = d_lap.LapNumber
            # print(d_lap.LapNumber)
            ref['Driver'] = d_laps.Driver.iloc[0]
            ref['DriverNumber'] = d_laps.DriverNumber.iloc[0]
            tel_df = pd.concat([tel_df, ref]).reset_index(drop=True)
"""
    #tel_df.to_csv('data/test_tel.csv', index=False)


def time_gap_race_all(year, race, drivers, num_laps=None, df_path=None):
    plotting.setup_mpl()
    fig, ax = plt.subplots()

    if df_path is None:
        race_dist = 3337
        laps_df = pd.DataFrame()

        session = ff1.get_session(year, race, 'R')
        session.load()

        for i in range(len(drivers)):
            d = drivers[i]
            d_laps = session.laps.pick_driver(d)
            if num_laps is None:
                laps = max(d_laps.LapNumber)
            else:
                laps = num_laps

            # use range(1, laps+1) to exclude lap 1
            for j in range(laps):
                if j < laps:
                    d_lap = d_laps[d_laps['LapNumber'] == (j + 1)].iloc[0]
                else:
                    d_lap = d_laps[d_laps['LapNumber'] == j].iloc[0]

                ref = d_lap.get_car_data(interpolate_edges=True)
                ref.add_driver_ahead()
                ref['LapNumber'] = d_lap.LapNumber
                ref['Driver'] = d_laps.Driver.iloc[0]
                ref['DriverNumber'] = d_laps.DriverNumber.iloc[0]
                """max_dist = max(ref['Distance'])
                ref_dist = race_dist * d_lap.LapNumber
                ref['Distance'] = ref['Distance'].apply(lambda row: (ref_dist / max_dist) * row)"""

                laps_df = pd.concat([laps_df, ref]).reset_index(drop=True)

            laps_df.to_csv('data/Monaco/Driver_Ahead.csv', index=False)


def main():
    cache(False)

    """array containing all drivers numbers for 2022 season"""
    drivers = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18', '6', '24', '22',
               '23', '47', '20']

    #time_gap_race_all(2022, 'Monaco', drivers)

    driver_ahead_test(2022, 'Monaco', ['NOR'])


if __name__ == '__main__':
    main()
