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

def create_target(df):
    df = df.sort_values(['DATE', 'STRIKE', 'OP_TYPE'])
    
    df['NEXT_LTP'] = df.groupby(['STRIKE', 'OP_TYPE'])['LTP'].shift(-1)
    
    df['RETURN'] = (df['NEXT_LTP'] - df['LTP']) / df['LTP']
    
    df['TARGET'] = (df['RETURN'] > 0).astype(int)
    
    df = df.dropna(subset=['NEXT_LTP', 'RETURN'])
    
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
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
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

def analyze_option_types(df):
    print("\n=== Analyzing Option Price Movements by Moneyness ===")
    
    atm_lower = 0.99
    atm_upper = 1.01

    otm_calls = df[(df['IS_CALL'] == 1) & (df['MONEYNESS'] < atm_lower)]
    print(f"\n--- 1% OTM CALLS (Underlying / STRIKE < {atm_lower}) ---")
    if not otm_calls.empty:
        print(f"Number of 1% OTM Calls: {len(otm_calls)}")
        print(f"Percentage of price increases (TARGET=1): {otm_calls['TARGET'].mean():.2%}")
    else:
        print("No 1% OTM Call options found for analysis with current thresholds.")

    otm_puts = df[(df['IS_CALL'] == 0) & (df['MONEYNESS'] > atm_upper)]
    print(f"\n--- 1% OTM PUTS (Underlying / STRIKE > {atm_upper}) ---")
    if not otm_puts.empty:
        print(f"Number of 1% OTM Puts: {len(otm_puts)}")
        print(f"Percentage of price increases (TARGET=1): {otm_puts['TARGET'].mean():.2%}")
    else:
        print("No 1% OTM Put options found for analysis with current thresholds.")
        
    atm_calls = df[(df['IS_CALL'] == 1) & (df['MONEYNESS'] >= atm_lower) & (df['MONEYNESS'] <= atm_upper)]
    print(f"\n--- 0.0% ATM CALLS ({atm_lower} <= Underlying / STRIKE <= {atm_upper}) ---")
    if not atm_calls.empty:
        print(f"Number of 0.0% ATM Calls: {len(atm_calls)}")
        print(f"Percentage of price increases (TARGET=1): {atm_calls['TARGET'].mean():.2%}")
    else:
        print("No 0.0% ATM Call options found for analysis with current thresholds.")

    atm_puts = df[(df['IS_CALL'] == 0) & (df['MONEYNESS'] >= atm_lower) & (df['MONEYNESS'] <= atm_upper)]
    print(f"\n--- 0.0% ATM PUTS ({atm_lower} <= Underlying / STRIKE <= {atm_upper}) ---")
    if not atm_puts.empty:
        print(f"Number of 0.0% ATM Puts: {len(atm_puts)}")
        print(f"Percentage of price increases (TARGET=1): {atm_puts['TARGET'].mean():.2%}")
    else:
        print("No 0.0% ATM Put options found for analysis with current thresholds.")

    itm_calls = df[(df['IS_CALL'] == 1) & (df['MONEYNESS'] > atm_upper)]
    print(f"\n--- 1% ITM CALLS (Underlying / STRIKE > {atm_upper}) ---")
    if not itm_calls.empty:
        print(f"Number of 1% ITM Calls: {len(itm_calls)}")
        print(f"Percentage of price increases (TARGET=1): {itm_calls['TARGET'].mean():.2%}")
    else:
        print("No 1% ITM Call options found for analysis with current thresholds.")

    itm_puts = df[(df['IS_CALL'] == 0) & (df['MONEYNESS'] < atm_lower)]
    print(f"\n--- 1% ITM PUTS (Underlying / STRIKE < {atm_lower}) ---")
    if not itm_puts.empty:
        print(f"Number of 1% ITM Puts: {len(itm_puts)}")
        print(f"Percentage of price increases (TARGET=1): {itm_puts['TARGET'].mean():.2%}")
    else:
        print("No 1% ITM Put options found for analysis with current thresholds.")


print("=== Simple Options Price Movement Predictor ===\n")

df = load_and_clean_data('NFO_DATA.csv')
if df is None:
    print("Failed to load data. Please check if 'NFO_DATA.csv' exists and is correctly formatted.")

print("Creating features...")
df = create_features(df)

print("Creating target variable...")
df = create_target(df)

print("Training model...")
model, X_test, y_test, y_pred = train_model(df)

analyze_option_types(df)
