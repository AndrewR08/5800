import fastf1 as ff1
import pandas as pd
from fastf1 import plotting
from fastf1 import utils
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

# options for easier readability on df print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
#pd.set_option("expand_frame_repr", False)


def cache(pc):
    if pc:
        # location of cache for pc
        ff1.Cache.enable_cache('D:/f1data')
    else:
        # location of cahce for mac
        ff1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def custom_delta(ref_lap, com_lap):
    ref = ref_lap.get_car_data(interpolate_edges=True).add_distance()
    comp = com_lap.get_car_data(interpolate_edges=True).add_distance()


def plot_delta(year, race, d1, d2, lap_num=8):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'R')
    session.load()
    d1_laps = session.laps.pick_driver(d1)
    d1_lap = d1_laps[d1_laps['LapNumber'] == lap_num].iloc[0]
    d2_laps = session.laps.pick_driver(d2)
    d2_lap = d2_laps[d2_laps['LapNumber'] == lap_num].iloc[0]
    print("--d1 lap--\n", d1_lap)
    print("\n--d2 lap--\n", d2_lap)

    d1_name = d1_lap.Driver
    d2_name = d2_lap.Drivera

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap)

    #print(ref_tel)
    #print(compare_tel)

    actual_gap = str(abs(d1_lap['LapTime'] - d2_lap['LapTime']))[14:]

    fig, ax = plt.subplots()
    # use telemetry returned by .delta_time for best accuracy,
    # this ensure the same applied interpolation and resampling
    ax.plot(ref_tel['Distance'], delta_time, '--', color='white')
    ax.set_ylabel("<-- " + d2_name + " ahead | " + d1_name + " ahead -->")
    plt.title('Lap Time Delta (Actual= ' + actual_gap + ' s)')
    plt.show()


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
    #ax.legend(['NOR', 'LEC'])
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
            d1_lap = d1_laps[d1_laps['LapNumber'] == (i+1)].iloc[0]
            ref = d1_lap.get_car_data(interpolate_edges=True).add_distance()
            ref = ref[['Time', 'Distance']]
            #print("---ref lap " + str(i+1) + "---\n", ref)

            d2_lap = d2_laps[d2_laps['LapNumber'] == (i+1)].iloc[0]
            comp = d2_lap.get_car_data(interpolate_edges=True).add_distance()
            comp = comp[['Time', 'Distance']]
            # print("\n---comp lap " + str(i+1) + "---\n", comp)

            if i != 0:
                ref_len = len(ref_laps)
                last_ref_dist = ref_laps['Distance'].iloc[ref_len-1]
                last_ref_time = ref_laps['Time'].iloc[ref_len-1]
                print("lap " + str(i) + ", " + str(last_ref_time))
                ref['Time'] = ref['Time'] + last_ref_time
                ref['Distance'] = ref['Distance'] + last_ref_dist

                comp_len = len(comp_laps)
                last_comp_dist = comp_laps['Distance'].iloc[comp_len-1]
                last_comp_time = comp_laps['Time'].iloc[comp_len-1]
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

        for j in range(laps):
            d_lap = d_laps[d_laps['LapNumber'] == (j+1)].iloc[0]
            ref = d_lap.get_car_data(interpolate_edges=True).add_distance()
            ref = ref[['Time', 'Distance']]

            if j != 0:
                ref_len = len(laps_df)
                last_ref_dist = laps_df['Distance'].iloc[ref_len-1]
                last_ref_time = laps_df['Time'].iloc[ref_len-1]
                ref['Time'] = ref['Time'] + last_ref_time
                ref['Distance'] = ref['Distance'] + last_ref_dist

            ref['LapNumber'] = d_lap.LapNumber
            ref['Driver'] = d_laps.Driver.iloc[0]
            ref['DriverNumber'] = d_laps.DriverNumber.iloc[0]
            laps_df = pd.concat([laps_df, ref])

        ax.plot(laps_df['Time'].loc[laps_df['Driver'] == d_laps.Driver.iloc[0]],
                laps_df['Distance'].loc[laps_df['Driver'] == d_laps.Driver.iloc[0]],
                color=plotting.driver_color(d_laps.Driver.iloc[0]), label=d_laps.Driver.iloc[0])

    # laps_df['Distance'] = laps_df['Distance'] / 1000
    # print(laps_df)
    # laps_df.to_csv('data/Monaco/MonacoTimeDist.csv', index=False)

    ax.set_xlabel("Time (h:mm)")
    ax.set_ylabel("Distance (m)")
    ax.legend(loc='center right')
    plt.title(race + " Lap Time vs Distance")
    plt.show()


def main():
    cache(False)

    drivers = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18', '6', '24', '22', '23',
               '47', '20']


    # drivers = ['VER', 'PER', 'LEC', 'SAI', 'RUS', 'HAM', 'NOR', 'RIC', 'ALO', 'OCO', 'BOT', 'ZHO', 'GAS', 'TSU',
    #            'STR', 'VET', 'MAG', 'MSC', 'ALB', 'LAT']

    # plot_delta(2022, 'Monaco', '4', '16', 32)
    # time_dist(2022, 'Monaco', '4', '16', 32)
    # time_dist_all(2022, 'Monaco', drivers, 8)

    # ---lap 7 OCO pass VET---
    # time_dist_race_two(2022, 'Monaco', ['OCO', 'VET'], 10)
    time_dist_race_all(2022, 'Monaco', drivers)


if __name__ == '__main__':
    main()
