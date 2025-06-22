import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_clean_data(filename):
    try:
        df = pd.read_csv(filename)
        print(f"Original data shape: {df.shape}")
        
        df['EXP_DATE'] = pd.to_datetime(df['EXP_DATE'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        
        df['DAYS_TO_EXPIRY'] = (df['EXP_DATE'] - df['DATE']).dt.days
        
        df = df[
            (df['LTP'] > 0) & 
            (df['STRIKE'] > 0) & 
            (df['ATM_CALL'] > 0) & 
            (df['ATM_PUT'] > 0) &
            (df['VOLUME'] >= 0) &
            (df['DAYS_TO_EXPIRY'] > 0) &
            (df['NCM_LTP'] > 0)
        ]
        
        key_columns = ['LTP', 'STRIKE', 'ATM_CALL', 'ATM_PUT', 'VOLUME', 'DAYS_TO_EXPIRY', 'NCM_LTP']
        df = df.dropna(subset=key_columns)
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_features(df):
    df['IS_CALL'] = (df['OP_TYPE'].str.upper().isin(['CE', 'CALL'])).astype(int)
    
    underlying_price = df['NCM_LTP'] 

    df['MONEYNESS'] = np.nan

    df.loc[df['IS_CALL'] == 1, 'MONEYNESS'] = underlying_price / df['STRIKE']
    df.loc[df['IS_CALL'] == 0, 'MONEYNESS'] = underlying_price / df['STRIKE']
    
    df['VOLUME_RATIO'] = df['VOLUME'] / (df['VOLUME'].rolling(window=10, min_periods=1).mean() + 1)
    
    return df

def create_target_and_triggers(df, price_change_threshold=0.02):
    df = df.sort_values(['DATE', 'STRIKE', 'OP_TYPE'])
    
    df['NEXT_LTP'] = df.groupby(['STRIKE', 'OP_TYPE'])['LTP'].shift(-1)
    
    df['RETURN'] = (df['NEXT_LTP'] - df['LTP']) / df['LTP']
    
    df['TARGET'] = 0
    df.loc[df['RETURN'] > price_change_threshold, 'TARGET'] = 1
    df.loc[df['RETURN'] < -price_change_threshold, 'TARGET'] = -1

    df = df.dropna(subset=['NEXT_LTP', 'RETURN', 'TARGET'])
    
    return df

def train_model(df):
    feature_columns = [
        'LTP', 'STRIKE', 'IS_CALL', 'DAYS_TO_EXPIRY',
        'MONEYNESS', 'VOLUME_RATIO'
    ]
    
    X = df[feature_columns].copy()
    y = df['TARGET'].copy()
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    return model, X_test, y_test, y_pred

def simulate_trades(df, model, X_test, y_test, holding_period_days=2, stop_loss_pct=0.05, take_profit_pct=0.10):
    print("\n=== Simulating Trades ===")
    
    test_df = X_test.copy()
    test_df['TRUE_TARGET'] = y_test
    
    test_indices = X_test.index
    sim_df = df.loc[test_indices].copy()
    
    sim_df['PREDICTED_TARGET'] = model.predict(X_test)

    trades = []
    
    sim_df = sim_df.sort_values(by=['DATE', 'STRIKE', 'OP_TYPE'])

    for index, row in sim_df.iterrows():
        prediction = row['PREDICTED_TARGET']
        current_ltp = row['LTP']
        true_return = row['RETURN']
        
        signal = None
        if prediction == 1:
            signal = 'LONG'
        elif prediction == -1:
            signal = 'SHORT' 
        
        if signal:
            trade_entry_price = current_ltp
            sl_price_long = trade_entry_price * (1 - stop_loss_pct)
            tp_price_long = trade_entry_price * (1 + take_profit_pct)
            
            sl_price_short = trade_entry_price * (1 + stop_loss_pct)
            tp_price_short = trade_entry_price * (1 - take_profit_pct)

            trade_exit_price = None
            trade_profit_loss = 0
            exit_reason = "Holding Period Expiry"
            
            actual_exit_price = trade_entry_price * (1 + true_return)

            if signal == 'LONG':
                if actual_exit_price <= sl_price_long:
                    trade_profit_loss = (sl_price_long - trade_entry_price) / trade_entry_price
                    exit_reason = "Stop Loss Hit"
                    trade_exit_price = sl_price_long
                elif actual_exit_price >= tp_price_long:
                    trade_profit_loss = (tp_price_long - trade_entry_price) / trade_entry_price
                    exit_reason = "Take Profit Hit"
                    trade_exit_price = tp_price_long
                else:
                    trade_profit_loss = true_return
                    trade_exit_price = actual_exit_price
                    
            elif signal == 'SHORT':
                if actual_exit_price >= sl_price_short:
                    trade_profit_loss = (trade_entry_price - sl_price_short) / trade_entry_price
                    exit_reason = "Stop Loss Hit"
                    trade_exit_price = sl_price_short
                elif actual_exit_price <= tp_price_short:
                    trade_profit_loss = (trade_entry_price - tp_price_short) / trade_entry_price
                    exit_reason = "Take Profit Hit"
                    trade_exit_price = tp_price_short
                else:
                    trade_profit_loss = -true_return
                    trade_exit_price = actual_exit_price

            trades.append({
                'DATE': row['DATE'],
                'STRIKE': row['STRIKE'],
                'OP_TYPE': row['OP_TYPE'],
                'SIGNAL': signal,
                'ENTRY_LTP': trade_entry_price,
                'EXIT_LTP': trade_exit_price,
                'PREDICTED_TARGET': prediction,
                'TRUE_RETURN': true_return,
                'PROFIT_LOSS_PCT': trade_profit_loss,
                'EXIT_REASON': exit_reason
            })
            
    trade_log_df = pd.DataFrame(trades)
    
    if not trade_log_df.empty:
        total_profit_loss = trade_log_df['PROFIT_LOSS_PCT'].sum()
        num_trades = len(trade_log_df)
        profitable_trades = (trade_log_df['PROFIT_LOSS_PCT'] > 0).sum()
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        
        print(f"\nSimulation Results:")
        print(f"Total Trades: {num_trades}")
        print(f"Total Portfolio Return (sum of individual trade returns): {total_profit_loss:.4f}")
        print(f"Profitable Trades: {profitable_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print("\nFirst 10 Trades:")
        print(trade_log_df.head(10))
        print("\nLast 10 Trades:")
        print(trade_log_df.tail(10))
        
        print("\nProfit/Loss by Exit Reason:")
        print(trade_log_df.groupby('EXIT_REASON')['PROFIT_LOSS_PCT'].mean().sort_values(ascending=False))
        
        print("\nProfit/Loss by Signal Type:")
        print(trade_log_df.groupby('SIGNAL')['PROFIT_LOSS_PCT'].mean().sort_values(ascending=False))

    else:
        print("No trades generated based on current predictions and thresholds.")
    
    return trade_log_df

print("=== Options Trading Strategy Backtester ===\n")

df = load_and_clean_data('NFO_DATA.csv')
if df is None:
    print("Failed to load data. Please check if 'NFO_DATA.csv' exists and is correctly formatted.")
    exit()

print("Creating features...")
df = create_features(df)

print("Creating target variable and trade triggers...")
df = create_target_and_triggers(df, price_change_threshold=0.01)

print("Training model...")
model, X_test, y_test, y_pred = train_model(df)

trade_log = simulate_trades(
    df, model, X_test, y_test, 
    holding_period_days=1,
    stop_loss_pct=0.03,
    take_profit_pct=0.05
)
