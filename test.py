import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('NFO_DATA.csv')

# Parse dates
df['EXP_DATE'] = pd.to_datetime(df['EXP_DATE'], errors='coerce')
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# Compute days to expiry (as int)
df['DAYSTOEXP'] = (df['EXP_DATE'] - df['DATE']).dt.days

# Basic cleaning
df = df[(df['LTP'] > 0) & (df['STRIKE'] > 0) & (df['ATM_CALL'] > 0) & (df['ATM_PUT'] > 0)]
df = df.dropna(subset=['LTP', 'STRIKE', 'ATM_CALL', 'ATM_PUT', 'VOLUME', 'DAYSTOEXP'])

print(f"Samples after cleaning: {len(df)}")

# Feature engineering
df['isCall'] = df['OP_TYPE'].apply(lambda x: 1 if str(x).upper() in ['CE', 'CALL'] else 0)
df['moneyness_ratio'] = df['LTP'] / df['STRIKE']
df['log_moneyness'] = np.log(df['moneyness_ratio'])
df['option_skew'] = df['ATM_CALL'] - df['ATM_PUT']
df['vol_normalized'] = df['VOLUME'].rolling(window=5, min_periods=1).mean()
df['vol_ratio'] = df['VOLUME'] / df['vol_normalized']

# Future return and label
df['next_LTP'] = df['LTP'].shift(-1)
df['Return'] = (df['next_LTP'] - df['LTP']) / df['LTP']
df['Target'] = df['Return'].apply(lambda x: 1 if x > 0 else 0)

# Drop NaNs only after all feature engineering
required_cols = ['LTP', 'STRIKE', 'ATM_CALL', 'ATM_PUT', 'VOLUME', 'next_LTP', 'DAYSTOEXP']
df = df.dropna(subset=required_cols)

print(f"Final samples for training: {len(df)}")

# Define features and target
features = ['LTP', 'STRIKE', 'isCall', 'DAYSTOEXP', 
            'moneyness_ratio', 'log_moneyness', 'option_skew', 'vol_ratio']

X = df[features]
y = df['Target']

# Train-test split (no shuffle to respect time order)

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(0)
X = X.clip(-1e5, 1e5)  # avoids float32 overflow



X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("NaNs in X_train:\n", X_train.isna().sum())
print("Infs in X_train:\n", np.isinf(X_train).sum())
print("Too large values in X_train:\n", (X_train.abs() > 1e6).sum())

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.4f}")
