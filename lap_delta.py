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
    d2_name = d2_lap.Driver

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


def time_dist(year, race, d1, d2):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'Q')
    session.load()
    d1_lap = session.laps.pick_driver(d1).pick_fastest()
    d2_lap = session.laps.pick_driver(d2).pick_fastest()


def main():
    cache(False)

    plot_delta(2022, 'Monaco', '4', '16', 32)


if __name__ == '__main__':
    main()
