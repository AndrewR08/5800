import fastf1
import numpy as np
import pandas as pd
import os
import requests
import json

# options for easier readability on df print
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_colwidth', None)
#pd.set_option("expand_frame_repr", False)


# **fastf1 api only available for 2018 season and later**
# 103 different races - ~60 laps/race - 20 drivers/race = 123,600 rows of data

def cache(pc):
    if pc:
        # location of cache for pc
        fastf1.Cache.enable_cache('D:/f1data')
    else:
        # location of cahce for mac
        fastf1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def format_circuits(circuit):
    if isinstance(circuit, str):
        circuit = circuit.replace(' ', '_')
    return circuit


def download_data():
    years = [2018, 2019, 2020, 2021, 2022]

    for y in years:
        schedule = fastf1.get_event_schedule(y)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        circuits = schedule.EventName
        circuits = circuits.apply(format_circuits)
        for c in circuits:
            race = fastf1.get_session(y, c, 'R')
            race.load()

            laps = race.laps
            path = "data/" + str(y)
            if not os.path.exists(path):
                os.makedirs(path)


def add_pitstop(df, filename):
    df['PitLap'] = df.loc[~df['PitInTime'].isna(), 'LapNumber']
    df.to_csv(filename, index=False)
    return df


def get_fastest_lap_data(year, name, s_type, driver=None):
    session = fastf1.get_session(year, name, s_type)
    session.load(telemetry=False)
    # get specific drivers fastest lap
    if driver is None:
        lap = session.laps.pick_fastest()
    else:
        lap = session.laps.pick_driver(driver).pick_fastest()
    return lap


def get_lap_weather(lap):
    return lap.get_weather_data()


def get_session_weather(year, name, s_type):
    session = fastf1.get_session(year, name, s_type)
    session.load(telemetry=False)
    return session.weather_data


def get_car_data(year, name, s_type):
    session = fastf1.get_session(year, name, s_type)
    session.load(telemetry=True)
    return session.car_data


def get_results(year, name, s_type):
    session = fastf1.get_session(year, name, s_type)
    session.load(telemetry=False)
    laps = session.laps
    results = pd.DataFrame(laps)
    """standings = standings.drop(['Time', 'LapTime', 'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                    'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL',
                    'SpeedST', 'IsPersonalBest', 'Compound', 'TyreLife', 'FreshTyre', 'Stint', 'LapStartTime',
                    'TrackStatus', 'IsAccurate'], axis=1)"""
    results = results.drop_duplicates(subset='DriverNumber', keep='last')
    return results


def get_standings(year, round):
    url = "https://ergast.com/api/f1/" + year + "/" + round + "/current/driverStandings.json"
    response = requests.get(url)
    data = response.json()
    drivers_standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']  # noqa: E501
    return drivers_standings


def main():
    # True for pc / False for mac
    cache(False)
    """file = 'data/Monaco/Monaco_Grand_Prix.csv'
    df = pd.read_csv('data/2022/Monaco_Grand_Prix.csv')
    df = add_pitstop(df, file)
    print(df)
    fl = get_fastest_lap_data(2022, 'Monaco', 'R', 'NOR')
    fl.to_csv('data/Monaco/NOR_FastestLap.csv')
    print(fl)

    weather = get_lap_weather(fl)
    weather.to_csv('data/Monaco/NOR_FL_Weather.csv')
    print(weather)

    s_weather = get_session_weather(2022, 'Monaco', 'R')
    weather.to_csv('data/Monaco/Weather.csv')
    print(s_weather)

    s_car_data = get_car_data(2022, 'Monaco', 'R')"""

    results = get_results(2022, 'Monaco', 'R')
    #results.to_csv('data/Monaco/Results.csv', index=False)

    # create method to add points for race results df
    # iterate through df and use list of points (18, 15, ...) to assign to drivers in results list
    points = [18, 15, 12, 10, 8, 6, 4, 2, 1]

    standings = get_standings('2022', '1')
    print(standings)


if __name__ == '__main__':
    main()
