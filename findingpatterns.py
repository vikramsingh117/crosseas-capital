import pandas as pd

def main():
    ohlc = pd.read_csv('SPOT_OHLC_FEATURES.csv', index_col=0, parse_dates=True)

    breakout = ohlc[ohlc['High_vs_Past3High'] > 0]
    print('Breakout days: Avg Next1DayReturn:', breakout['Next1DayReturn'].mean())

    large_intraday = ohlc[ohlc['Return'] > 0.01]
    print('Large intraday return (>1%): Avg Next1DayReturn:', large_intraday['Next1DayReturn'].mean())

    ohlc['Past3Up'] = (ohlc['Return'].shift(1) > 0) & (ohlc['Return'].shift(2) > 0) & (ohlc['Return'].shift(3) > 0)
    reversal = ohlc[(ohlc['Past3Up']) & (ohlc['Return'] < 0)]
    print('Reversal after 3-day uptrend: Avg Next1DayReturn:', reversal['Next1DayReturn'].mean())

if __name__ == '__main__':
    main() 