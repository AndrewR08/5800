import fastf1 as ff1
import pandas as pd
from fastf1 import plotting
from fastf1 import utils
from matplotlib import pyplot as plt
import numpy as np
import datetime
import math
from scipy import stats


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


def plot_delta(year, race, d1, d2, lap_num=8):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'R')
    session.load()
    d1_laps = session.laps.pick_driver(d1)
    d1_lap = d1_laps[d1_laps['LapNumber'] == lap_num].iloc[0]
    d2_laps = session.laps.pick_driver(d2)
    d2_lap = d2_laps[d2_laps['LapNumber'] == lap_num].iloc[0]
    # print("--d1 lap--\n", d1_lap)
    # print("\n--d2 lap--\n", d2_lap)

    d1_name = d1_lap.Driver
    d2_name = d2_lap.Driver

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap)

    print(ref_tel[['Time', 'Distance']])
    print(compare_tel[['Time', 'Distance']])
    print(delta_time)

    actual_gap = str(abs(d1_lap['LapTime'] - d2_lap['LapTime']))[14:]

    fig, ax = plt.subplots()
    # use telemetry returned by .delta_time for best accuracy,
    # this ensure the same applied interpolation and resampling
    ax.plot(delta_time, ref_tel['Distance'], '--', color='white')
    ax.set_xlabel("<-- " + d2_name + " ahead | " + d1_name + " ahead -->")
    ax.set_ylabel("Distance")
    plt.title('Lap Time Delta (Actual= ' + actual_gap + ' s)')
    # plt.show()


def time_dist(year, race, d1, d2, lap_num=8):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'R')
    session.load()
    d1_laps = session.laps.pick_driver(d1)
    d1_lap = d1_laps[d1_laps['LapNumber'] == lap_num].iloc[0]
    d2_laps = session.laps.pick_driver(d2)
    d2_lap = d2_laps[d2_laps['LapNumber'] == lap_num].iloc[0]
    """print("--d1 lap--\n", d1_lap)
    print("\n--d2 lap--\n", d2_lap)"""

    d1_name = d1_lap.Driver
    d2_name = d2_lap.Driver

    ref = d1_lap.get_car_data(interpolate_edges=True).add_distance()
    print(ref.columns)
    print("---ref---\n", ref[['Time', 'Distance']])
    comp = d2_lap.get_car_data(interpolate_edges=True).add_distance()
    print("\n---comp---\n", comp[['Time', 'Distance']])

    fig, ax = plt.subplots()
    ax.plot(ref['Time'], ref['Distance'], color=plotting.team_color(d1_lap['Team']), label=d1_name)
    ax.plot(comp['Time'], comp['Distance'], color=plotting.team_color(d2_lap['Team']), label=d2_name)
    ax.set_xlabel("Time (m:ss)")
    ax.set_ylabel("Distance (m)")
    ax.legend()
    plt.title(d1_name + "/" + d2_name + " Lap Time vs Distance (Lap " + str(lap_num) + ")")
    plt.show()


def time_dist_all(year, race, drivers, lap_num=8):
    plotting.setup_mpl()
    fig, ax = plt.subplots()

    session = ff1.get_session(year, race, 'R')
    session.load()

    for i in range(len(drivers)):
        laps = session.laps.pick_driver(drivers[i])
        lap = laps[laps['LapNumber'] == lap_num].iloc[0]

        ref = lap.get_car_data(interpolate_edges=True).add_distance()
        ax.plot(ref['Time'], ref['Distance'], color=plotting.driver_color(lap['Driver']), label=lap['Driver'])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Time (m:ss)")
    ax.set_ylabel("Distance (m)")
    # ax.legend(['NOR', 'LEC'])
    plt.title("Lap Time vs Distance (Lap " + str(lap_num) + ")")
    plt.tight_layout()
    plt.show()


def time_dist_race_two(year, race, drivers, num_laps=0):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'R')
    session.load()

    d1 = drivers[0]
    d2 = drivers[1]

    d1_laps = session.laps.pick_driver(d1)
    d2_laps = session.laps.pick_driver(d2)

    d1_name = d1_laps.Driver.iloc[0]
    d2_name = d2_laps.Driver.iloc[0]

    ref_laps = pd.DataFrame()
    comp_laps = pd.DataFrame()

    if num_laps == 0:
        num_laps = min(len(d1_laps), len(d2_laps))

    # when driver 2 has more total laps than driver 2 (or equal laps)
    if len(d1_laps) <= len(d2_laps):
        for i in range(num_laps):
            d1_lap = d1_laps[d1_laps['LapNumber'] == (i + 1)].iloc[0]
            ref = d1_lap.get_car_data(interpolate_edges=True).add_distance()
            ref = ref[['Time', 'Distance']]
            # print("---ref lap " + str(i+1) + "---\n", ref)

            d2_lap = d2_laps[d2_laps['LapNumber'] == (i + 1)].iloc[0]
            comp = d2_lap.get_car_data(interpolate_edges=True).add_distance()
            comp = comp[['Time', 'Distance']]
            # print("\n---comp lap " + str(i+1) + "---\n", comp)

            if i != 0:
                ref_len = len(ref_laps)
                last_ref_dist = ref_laps['Distance'].iloc[ref_len - 1]
                last_ref_time = ref_laps['Time'].iloc[ref_len - 1]
                print("lap " + str(i) + ", " + str(last_ref_time))
                ref['Time'] = ref['Time'] + last_ref_time
                ref['Distance'] = ref['Distance'] + last_ref_dist

                comp_len = len(comp_laps)
                last_comp_dist = comp_laps['Distance'].iloc[comp_len - 1]
                last_comp_time = comp_laps['Time'].iloc[comp_len - 1]
                comp['Time'] = comp['Time'] + last_comp_time
                comp['Distance'] = comp['Distance'] + last_comp_dist
            """else:
                last_ref_dist = ref['Distance'].iloc[-1]
                last_ref_time = ref['Time'].iloc[-1]
                print("lap 1, " + str(last_ref_time))"""

            ref_laps = pd.concat([ref_laps, ref])
            comp_laps = pd.concat([comp_laps, comp])

    print("lap " + str(num_laps) + ", " + str(ref_laps['Time'].iloc[-1]))
    print()
    print(ref_laps)
    print(comp_laps)

    ref_laps['Distance'] = ref_laps['Distance'] / 1000
    comp_laps['Distance'] = comp_laps['Distance'] / 1000

    fig, ax = plt.subplots()
    ax.plot(ref_laps['Time'], ref_laps['Distance'], color=plotting.team_color(d1_lap['Team']), label=d1_name)
    ax.plot(comp_laps['Time'], comp_laps['Distance'], color=plotting.team_color(d2_lap['Team']), label=d2_name)
    ax.set_xlabel("Time (h:mm)")
    ax.set_ylabel("Distance (km)")
    ax.legend(loc='center right')
    plt.title(d1_name + "/" + d2_name + " Lap Time vs Distance")
    plt.show()


def time_dist_race_all(year, race, drivers, num_laps=None):
    plotting.setup_mpl()
    fig, ax = plt.subplots()

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
            # for j in range(laps):
            d_lap = d_laps[d_laps['LapNumber'] == (j + 1)].iloc[0]
            ref = d_lap.get_car_data(interpolate_edges=True).add_distance()
            ref = ref[['Time', 'Distance']]

            # use j != 1 to exclude lap 1
            if j != 0:
                # if j != 0:
                ref_len = len(laps_df)
                last_ref_dist = laps_df['Distance'].iloc[ref_len - 1]
                last_ref_time = laps_df['Time'].iloc[ref_len - 1]
                ref['Time'] = ref['Time'] + last_ref_time
                ref['Distance'] = ref['Distance'] + last_ref_dist

            ref['LapNumber'] = d_lap.LapNumber
            ref['Driver'] = d_laps.Driver.iloc[0]
            ref['DriverNumber'] = d_laps.DriverNumber.iloc[0]
            ref['MaxDistance'] = max(ref['Distance'])

            laps_df = pd.concat([laps_df, ref]).reset_index(drop=True)

        ax.plot(laps_df['Time'].loc[laps_df['Driver'] == d_laps.Driver.iloc[0]],
                laps_df['Distance'].loc[laps_df['Driver'] == d_laps.Driver.iloc[0]],
                color=plotting.driver_color(d_laps.Driver.iloc[0]), label=d_laps.Driver.iloc[0])

    # print(laps_df)
    # laps_df.to_csv('data/Monaco/MonacoTD_LEC_SAI_L1_TEST.csv', index=False)

    ax.set_xlabel("Time (h:mm)")
    ax.set_ylabel("Distance (m)")
    ax.legend(loc='center right')
    plt.title(race + " " + str(year) + " Lap Time vs Distance Gap")
    plt.show()


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
            #print("*****"+d_laps.Driver.iloc[0]+"*****\n")
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
                #if j == 1: ref.to_csv('data/Monaco/RAW_REF2.csv', index=False)
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
        """plot distribution of time difference intervals in histogram"""
        # laps_df['TimeDiff'].plot(kind='hist', edgecolor='black', xticks=[0, 0.125, 0.25, 0.375, 0.5, 0.75, 1])

        # print(laps_df)
        laps_df.to_csv('data/Monaco/MonacoTD_lapsdf_TEST3.csv', index=False)

        """ *** max time = 0 days 01:58:30.069000 // 7110.069s *** """
        max_time = math.ceil(max(laps_df['Time']).total_seconds())
        print(max_time)
        laps_df_new = pd.DataFrame()
        t = np.linspace(0, max_time, math.floor(max_time / 0.25))  # use 0.18s sample rate
        laps_df_new['Time'] = t

        for d in drivers:
            tp = laps_df['Time'].loc[laps_df['DriverNumber'] == d].apply(lambda row: row.total_seconds())
            dp = laps_df['Distance'].loc[laps_df['DriverNumber'] == d]
            d_new = np.interp(t, tp, dp)
            laps_df_new['Distance_' + str(laps_df.Driver.loc[laps_df.DriverNumber == d].iloc[0])] = d_new

        df = laps_df_new

    else:
        df = pd.read_csv(df_path)

    # df['Time'] = df['Time'].apply(lambda row: datetime.timedelta(seconds=row))
    leader = str(df.columns[1])[9:]
    #print(leader)
    for i in range(len(drivers)):
        d = str(df.columns[i + 1])[9:]
        if d != leader:
            df['DistanceGap_' + d] = df['Distance_' + leader] - df['Distance_' + d]
            # df['DistanceGap_'+d] = df['DistanceGap_'+d].loc[df['DistanceGap_'+d] > 3337].apply(lambda row: 0)
            #df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
            ax.plot(df['Time'].apply(lambda row: datetime.timedelta(seconds=row)), df['DistanceGap_' + d],
                    color=plotting.driver_color(d),
                    label=d)
    df.to_csv('data/Monaco/__GAP1__.csv', index=False)

    #ax.set_ylim(bottom= -3337, top=3337)
    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Distance Gap (m)")
    ax.legend(loc='lower center',
              ncol=4, fancybox=True, shadow=True)
    plt.title(race + " " + str(year) + " Time vs Distance Gap to " + leader)
    plt.show()

    #df.to_csv('data/Monaco/MonacoTD_TEST3.csv', index=False)


def main():
    cache(False)

    """array containing all drivers numbers for 2022 season"""
    drivers = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18', '6', '24', '22', '23',
               '47', '20']

    # remove last 6 drivers for better comparison -- fix --
    # drivers = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18']

    # drivers = ['VER', 'PER', 'LEC', 'SAI', 'RUS', 'HAM', 'NOR', 'RIC', 'ALO', 'OCO', 'BOT', 'ZHO', 'GAS', 'TSU',
    #            'STR', 'VET', 'MAG', 'MSC', 'ALB', 'LAT']

    # plot_delta(2022, 'Monaco', '16', '55', 2)

    """graph two drivers time vs distance for a single lap"""
    # time_dist(2022, 'Monaco', '16', '55', 2)

    """graph all drivers time vs distance for a single lap """
    # time_dist_all(2022, 'Monaco', drivers, 8)

    """---lap 7 OCO pass VET---
    graph two drivers time vs distance for entire race"""
    # time_dist_race_two(2022, 'Monaco', ['OCO', 'VET'], 10)

    """graph all drivers time vs distance for entire race"""
    # time_dist_race_all(2022, 'Monaco', drivers)

    time_gap_race_all(2022, 'Monaco', ['16', '55'], num_laps=1) #, num_laps=4) #, df_path='data/Monaco/__TEST__.csv')


if __name__ == '__main__':
    main()
