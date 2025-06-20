import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    df = pd.read_csv('SPOT_OHLC_FEATURES.csv', index_col=0, parse_dates=True)

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    features = ['Return', 'High_vs_Past3High', 'MA_5']
    
    df.dropna(inplace=True)

    X = df[features]
    y = df['Target']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    print("\nModel Evaluation on Test Data:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Down', 'Up']))
    
    print("\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index=features)
    print(importances.sort_values(ascending=False))


if __name__ == '__main__':
    main() 