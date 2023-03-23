import fastf1 as ff1
import pandas as pd
from fastf1 import plotting
from fastf1 import utils
from matplotlib import pyplot as plt


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


def time_dist_race(year, race, drivers, num_laps=1):
    plotting.setup_mpl()

    d1 = drivers[0]
    d2 = drivers[1]

    lap_num = 5

    session = ff1.get_session(year, race, 'R')
    session.load()
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
                ref['Time'] = ref['Time'] + last_ref_time
                ref['Distance'] = ref['Distance'] + last_ref_dist

                comp_len = len(comp_laps)
                last_comp_dist = comp_laps['Distance'].iloc[comp_len-1]
                last_comp_time = comp_laps['Time'].iloc[comp_len-1]
                comp['Time'] = comp['Time'] + last_comp_time
                comp['Distance'] = comp['Distance'] + last_comp_dist

            ref_laps = pd.concat([ref_laps, ref])
            comp_laps = pd.concat([comp_laps, comp])

    # when driver 1 has more total laps than driver 2
    else:
        for i in range(len(d2_laps)):
            d1_lap = d1_laps[d1_laps['LapNumber'] == lap_num].iloc[i]
            d2_lap = d2_laps[d2_laps['LapNumber'] == lap_num].iloc[i]

    print(ref_laps)
    print(comp_laps)

    ref_laps['Distance'] = ref_laps['Distance'] / 1000
    comp_laps['Distance'] = comp_laps['Distance'] / 1000

    fig, ax = plt.subplots()
    ax.plot(ref_laps['Time'], ref_laps['Distance'], color=plotting.team_color(d1_lap['Team']), label=d1_name)
    ax.plot(comp_laps['Time'], comp_laps['Distance'], color=plotting.team_color(d2_lap['Team']), label=d2_name)
    ax.set_xlabel("Time (h:mm)")
    ax.set_ylabel("Distance (km)")
    ax.legend()
    plt.title(d1_name + "/" + d2_name + " Lap Time vs Distance")
    plt.show()


def main():
    cache(False)

    drivers = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18', '6', '24', '22', '23',
               '47', '20']

    #plot_delta(2022, 'Monaco', '4', '16', 32)
    #time_dist(2022, 'Monaco', '4', '16', 32)
    #time_dist_all(2022, 'Monaco', drivers, 8)
    #lap 7 oco pass vet
    time_dist_race(2022, 'Monaco', ['OCO', 'VET'], 10)


if __name__ == '__main__':
    main()
