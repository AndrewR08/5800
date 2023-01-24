import fastf1
import pandas as pd

# options for easier readability on df print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option("expand_frame_repr", False)

#location of cache for data
fastf1.Cache.enable_cache('D:/f1data')

year = 2022
circuit = 'Miami'

race = fastf1.get_session(year, circuit, 'R')
race.load()

laps = race.laps
ver = race.get_driver('VER')
lec = race.get_driver('LEC')
#print(laps)

ver_laps = laps.pick_driver('VER')
lec_laps = laps.pick_driver('LEC')
print(laps)
print()
"""print(lec_laps)
print(ver_laps.get_telemetry())
print()
print(lec_laps.get_telemetry())"""




