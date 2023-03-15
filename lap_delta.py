import fastf1 as ff1
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
    ax.plot(ref['Time'], ref['Distance'], color=plotting.team_color(d1_lap['Team']))
    ax.plot(comp['Time'], comp['Distance'], color=plotting.team_color(d2_lap['Team']))
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance")
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
        ax.plot(ref['Time'], ref['Distance'], color=plotting.team_color(lap['Team']))

    ax.set_xlabel("Time")
    ax.set_ylabel("Distance")
    plt.title("Lap Time vs Distance (Lap " + str(lap_num) + ")")
    plt.show()


def main():
    cache(False)

    drivers = ['11', '55', '1', '16', '63', '4', '14', '44', '77', '5', '10', '31', '3', '18', '6', '24', '22', '23',
               '47', '20']

    #plot_delta(2022, 'Monaco', '4', '16', 32)
    time_dist(2022, 'Monaco', '4', '16', 32)
    time_dist_all(2022, 'Monaco', drivers, 8)


if __name__ == '__main__':
    main()
