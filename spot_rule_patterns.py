import pandas as pd

def main():
    ohlc = pd.read_csv('SPOT_OHLC_FEATURES.csv', index_col=0, parse_dates=True)
    rules = []

    # Rule 1: Breakout day (High_vs_Past3High > 0)
    breakout = ohlc[ohlc['High_vs_Past3High'] > 0].copy()
    breakout['Rule'] = 'Breakout'
    rules.append(breakout)

    # Rule 2: Large positive intraday return (Return > 1%)
    large_up = ohlc[ohlc['Return'] > 0.01].copy()
    large_up['Rule'] = 'LargeUpDay'
    rules.append(large_up)

    # Rule 3: Large negative intraday return (Return < -1%)
    large_down = ohlc[ohlc['Return'] < -0.01].copy()
    large_down['Rule'] = 'LargeDownDay'
    rules.append(large_down)


    # Rule 5: Inside day (High < previous High and Low > previous Low)
    ohlc['PrevHigh'] = ohlc['High'].shift(1)
    ohlc['PrevLow'] = ohlc['Low'].shift(1)
    inside = ohlc[(ohlc['High'] < ohlc['PrevHigh']) & (ohlc['Low'] > ohlc['PrevLow'])].copy()
    inside['Rule'] = 'InsideDay'
    rules.append(inside)

    # Combine all rule-matched rows
    all_rules = pd.concat(rules)
    all_rules = all_rules.sort_index()

    # Export to CSV
    all_rules.to_csv('SPOT_RULE_MATCHES.csv')
    print('Exported rule-matched rows to SPOT_RULE_MATCHES.csv')

if __name__ == '__main__':
    main() 