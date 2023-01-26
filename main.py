import fastf1
import pandas as pd
import os

# options for easier readability on df print
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
# pd.set_option("expand_frame_repr", False)


# **fastf1 api only available for 2018 season and later**

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


# True for pc / False for mac
cache(False)

years = [2018, 2019, 2020, 2021, 2022]

"""for y in years:
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

df = pd.read_csv('data/2022/United_States_Grand_Prix.csv')
print(df)
