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


def plot_delta(year, race, d1, d2):
    plotting.setup_mpl()

    session = ff1.get_session(year, race, 'Q')
    session.load()
    d1_lap = session.laps.pick_driver(d1).pick_fastest()
    d2_lap = session.laps.pick_driver(d2).pick_fastest()

    d1_name = d1_lap.Driver
    d2_name = d2_lap.Driver

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap)
    # ham is reference, lec is compared

    print(ref_tel)

    fig, ax = plt.subplots()
    # use telemetry returned by .delta_time for best accuracy,
    # this ensure the same applied interpolation and resampling
    ax.plot(ref_tel['Distance'], ref_tel['Speed'],
            color=plotting.team_color(d1_lap['Team']))
    ax.plot(compare_tel['Distance'], compare_tel['Speed'],
            color=plotting.team_color(d2_lap['Team']))

    twin = ax.twinx()
    twin.plot(ref_tel['Distance'], delta_time, '--', color='white')
    twin.set_ylabel("<-- " + d2_name + " ahead | " + d1_name + " ahead -->")
    plt.title('Lap Time Delta')
    plt.show()


def main():
    cache(False)

    plot_delta(2022, 'Monaco', '4', '55')


if __name__ == '__main__':
    main()