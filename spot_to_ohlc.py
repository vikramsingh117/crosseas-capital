import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('SPOT_DATA.csv')

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    df.set_index('TIMESTAMP', inplace=True)

    ohlc = df['NCM_LTP1'].resample('1D').agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['Open', 'High', 'Low', 'Close']

    ohlc = ohlc.dropna(how='all')

    # Feature current day return
    ohlc['Return'] = (ohlc['Close'] - ohlc['Open']) / ohlc['Open']
    # Feature two: current high and past 3-day rolling hig
    ohlc['Past3DayHigh'] = ohlc['High'].shift(1).rolling(window=3).max()
    ohlc['High_vs_Past3High'] = ohlc['High'] - ohlc['Past3DayHigh']

    # Feature 5: Next day  returns
    ohlc['Next1DayReturn'] = ohlc['Close'].shift(-1) / ohlc['Close'] - 1
    ohlc['Next2DayReturn'] = ohlc['Close'].shift(-2) / ohlc['Close'] - 1





    ohlc.to_csv('SPOT_OHLC_FEATURES.csv')
    print('Saved daily OHLC with features to SPOT_OHLC_FEATURES.csv')

if __name__ == '__main__':
    main()