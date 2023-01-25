import fastf1
import pandas as pd
import os

# options for easier readability on df print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option("expand_frame_repr", False)


# **fastf1 api only available for 2018 season and later**

#location of cache for data
fastf1.Cache.enable_cache('D:/f1data')

years = [2018, 2019, 2020, 2021, 2022]
circuits = ['Bahrain', 'Saudi_Arabia', 'Australia', 'Imola', 'Miami', 'Spain', 'Monaco', 'Azerbaijan', 'Canada',
            'Great_Britain', 'Austria', 'France', 'Hungary', 'Belgium', 'Netherlands', 'Monza', 'Singapore', 'Japan',
            'Aramco_United_States', 'Mexico', 'Brazil', 'Abu_Dhabi']

schedule = fastf1.get_event_schedule(2017)
print(schedule)

"""for y in years:
    for c in circuits:
        race = fastf1.get_session(y, c, 'R')
        race.load()

        laps = race.laps
        path = "data/" + str(y)
        if not os.path.exists(path):
            os.makedirs(path)

        file = path + "/" + c + ".csv"
        print(file)
        laps.to_csv(file, index=False)"""

"""ver = race.get_driver('VER')
lec = race.get_driver('LEC')
#print(laps)

ver_laps = laps.pick_driver('VER')
lec_laps = laps.pick_driver('LEC')
#print(laps)
print()
print(ver_laps)
print(ver_laps.get_telemetry())
#print()
#print(lec_laps.get_telemetry())"""


"""race = fastf1.get_session(2018, 'Monaco', 'R')
race.load()
laps = race.laps
print(laps)"""