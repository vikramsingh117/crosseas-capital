import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ohlc = pd.read_csv('SPOT_OHLC_FEATURES.csv', index_col=0, parse_dates=True)

    sns.set_style('whitegrid')

    plt.figure(figsize=(10, 6))
    sns.histplot(ohlc['Return'], bins=50, kde=True, color='blue')
    plt.title(' Daily Spot Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig('visuals/spotreturn.png')
    plt.close()

    # 2. Breakout vs. Next Day Return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='High_vs_Past3High', y='Next1DayReturn', data=ohlc, alpha=0.5)
    plt.title('BreakoutVS Next Day Return')
    plt.xlabel('High vs.,, last 3 dayHigh(Breakout)')
    plt.ylabel('Next 1-Day Return')
    plt.axhline(0, color='red', linestyle='--')
    plt.savefig('visuals/breakout_vs_next_return.png')
    plt.close()


if __name__ == '__main__':
    main() 