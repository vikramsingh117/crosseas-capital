import pandas as pd

def main():
    df = pd.read_csv('SPOT_DATA.csv')

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    df.set_index('TIMESTAMP', inplace=True)

    ohlc = df['NCM_LTP1'].resample('1D').agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['Open', 'High', 'Low', 'Close']

    ohlc = ohlc.dropna(how='all')

    ohlc.to_csv('SPOT_OHLC.csv')
    print('Saved daily OHLC to SPOT_OHLC.csv')