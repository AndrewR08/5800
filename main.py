import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.pyplot import figure
import numpy as np
import fastf1
from fastf1 import plotting


def plt_pit():
    df = pd.read_csv('data/2022/United_States_Grand_Prix_Fixed.csv')
    print(df)

    print(len(df['PitLap'].unique()))
    x = df['PitLap'].unique()
    y = df['Driver'].unique()
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    loc = plticker.MultipleLocator(base=5.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    plt.xlabel('Lap Number')
    plt.ylabel('Driver')
    plt.title('Driver Pit Stops by Lap')
    plt.show()


def ff1_pit():
    # location of cache for data
    fastf1.Cache.enable_cache('D:/f1data')

    year = 2022
    circuit = 'Miami'

    # Load the session data
    race = fastf1.get_session(year, circuit, 'R')
    laps = race.load_laps(with_telemetry=True)

    driver_stints = laps[['Driver', 'Stint', 'Compound', 'LapNumber']].groupby(
        ['Driver', 'Stint', 'Compound']
    ).count().reset_index()

    driver_stints = driver_stints.rename(columns={'LapNumber': 'StintLength'})

    driver_stints = driver_stints.sort_values(by=['Stint'])

    compound_colors = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFF200',
        'HARD': '#EBEBEB',
        'INTERMEDIATE': '#39B54A',
        'WET': '#00AEEF',
    }

    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    for driver in race.results['Abbreviation']:
        stints = driver_stints.loc[driver_stints['Driver'] == driver]

        previous_stint_end = 0
        for _, stint in stints.iterrows():
            plt.barh(
                [driver],
                stint['StintLength'],
                left=previous_stint_end,
                color=compound_colors[stint['Compound']],
                edgecolor="black"
            )

            previous_stint_end = previous_stint_end + stint['StintLength']

    # Set title
    plt.title(f'Race strategy - {circuit} {year}')

    # Set x-label
    plt.xlabel('Lap')

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Remove frame from plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.show()


def main():
    plt_pit()
    ff1_pit()


if __name__ == '__main__':
    main()