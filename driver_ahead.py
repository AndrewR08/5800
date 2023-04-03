import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import matplotlib.pyplot as plt

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


def driver_ahead():
    ff1.plotting.setup_mpl()

    session = ff1.get_session(2022, 'Monaco', 'R')
    session.load()

    DRIVER = 'SAI'  # which driver; need to specify number and abbreviation
    LAP_N = 18      # which lap number to plot

    drv_laps = session.laps.pick_driver(DRIVER)
    drv_lap = drv_laps[(drv_laps['LapNumber'] == LAP_N)]  # select the lap

    # create a matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot()

    # ############### new
    df_new = drv_lap.get_car_data().add_driver_ahead()
    df_new['DistanceToDriverAhead'] = df_new['DistanceToDriverAhead'].fillna(0)
    print(df_new)
    ax.plot(df_new['Time'], df_new['DistanceToDriverAhead'],  color=plotting.driver_color(DRIVER), label=DRIVER)

    """# ############### legacy
    df_legacy = fastf1.legacy.inject_driver_ahead(session)[DRIVER_NUMBER].slice_by_lap(drv_lap)
    ax.plot(df_legacy['Time'], df_legacy['DistanceToDriverAhead'], label='legacy')"""

    plt.legend()
    plt.show()


def main():
    cache(False)
    driver_ahead()


if __name__ == '__main__':
    main()
