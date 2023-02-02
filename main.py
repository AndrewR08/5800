import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def main():
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


if __name__ == '__main__':
    main()