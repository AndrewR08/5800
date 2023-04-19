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


def animate(year, race, drivers, lap_num=8):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'R')
    session.load()

    laps_df = pd.DataFrame()

    for i in range(len(drivers)):
        d = drivers[i]
        d_laps = session.laps.pick_driver(d)
        d_lap = d_laps[d_laps['LapNumber'] == lap_num].iloc[0]
        ref = d_lap.get_pos_data(interpolate_edges=True)
        ref['LapNumber'] = d_lap.LapNumber
        ref['Driver'] = d_laps.Driver.iloc[0]
        ref['DriverNumber'] = d_laps.DriverNumber.iloc[0]

        laps_df = pd.concat([laps_df, ref]).reset_index(drop=True)
        
        max_time = math.ceil(max(laps_df['Time']).total_seconds())

    """fig, ax = plt.subplots()
    # use telemetry returned by .delta_time for best accuracy,
    # this ensure the same applied interpolation and resampling
    ax.plot(delta_time, ref_tel['Distance'], '--', color='white')
    ax.set_xlabel("<-- " + d2_name + " ahead | " + d1_name + " ahead -->")
    ax.set_ylabel("Distance")
    plt.title('Lap Time Delta (Actual= ' + actual_gap + ' s)')
    # plt.show()"""


def main():
    cache(True)
    animate(2022, 'Monaco', 'LEC', 'SAI', lap_num=2)


if __name__ == '__main__':
    main()

