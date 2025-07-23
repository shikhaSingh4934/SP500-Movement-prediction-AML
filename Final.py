# market_movement_prediction.py
# This script loads multiple financial datasets, merges them, performs EDA,
# feature engineering, and trains classification models to predict daily movement.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_and_preprocess_data(file_paths):
    """Load CSV files, fill missing values, parse dates, and set Date as index."""
    dfs = {key: pd.read_csv(path) for key, path in file_paths.items()}

    # Fill missing values forward and backward
    for key, df in dfs.items():
        dfs[key] = df.fillna(method='ffill').fillna(method='bfill')

    # Convert Date column to datetime and set as index
    for key, df in dfs.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    return dfs


def merge_datasets(dfs):
    """Merge multiple dataframes on their datetime index."""
    combined_df = dfs['sp500_df']
    # Join others on index, add suffix to avoid column conflicts
    for key in ['gold_df', 'crudeoil_df', 'eur_df', 'gbp_df', 'cny_df', 'jpy_df', 'usidx_df']:
        combined_df = combined_df.join(dfs[key], how='inner', rsuffix=f'_{key}')
    return combined_df


def clean_numeric_columns(df):
    """Clean string columns containing commas, percentages, or units and convert to numeric."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col].str.replace(',', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.replace('M', 'e6', regex=False)
                .str.replace('B', 'e9', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward fill any remaining NaNs
    df.fillna(method='ffill', inplace=True)


def perform_eda(df):
    """Perform basic exploratory data analysis."""
    print(df.describe())

    # Histograms for numerical features
    df.hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Time series plot of S&P 500 closing price
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'])
    plt.title('S&P 500 Futures Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()


def feature_engineering(df):
    """Create momentum, moving averages, volatility, ratio, and RSI features."""
    df['Momentum_10d'] = df['Price'] - df['Price'].shift(10)
    df['Momentum_5d'] = df['Price'] - df['Price'].shift(5)
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_5'] = df['Price'].rolling(window=5).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    df['Price_MA5'] = df['Price'].rolling(window=5).mean()
    df['Price_volatility_7d'] = df['Price'].rolling(window=7).std()

    # Ratios of other assets to S&P 500 price
    df['Gold_to_SP500'] = df['Price_gold_df'] / df['Price']
    df['DollarIndex_to_SP500'] = df['Price_usidx_df'] / df['Price']

    # RSI calculation
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))


def create_binary_target(df):
    """Create binary target column indicating daily price movement direction."""
    df['Binary Movement'] = 0
    df.loc[df['Price'].diff() > 0, 'Binary Movement'] = 1
    df.loc[df['Price'].diff() < 0, 'Binary Movement'] = -1


def drop_unnecessary_columns(df):
    """Drop specified columns to reduce overfitting."""
    drop_columns = ['Open', 'High', 'Low', 'Vol.', 'Change %', 'Price']
    cols_to_drop = [col for col in df.columns if any(suffix in col for suffix in drop_columns)]
    df.drop(columns=cols_to_drop, inplace=True)


def train_models(X_train, y_train, X_test, y_test):
    """Train models with different scalers and hyperparameters, return results."""
    scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }

    models = {
        'Logistic Regression': (LogisticRegression(solver='saga', max_iter=1000), {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l1', 'l2']
        }),
        'Linear SVM': (LinearSVC(max_iter=1000), {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l1', 'l2']
        }),
        'Gradient Boosting': (GradientBoostingClassifier(), {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5]
        })
    }

    results = []

    for model_name, (model, param_grid) in models.items():
        best_scaler = None
        best_score = -np.inf

        for scaler_name, scaler in scalers.items():
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model)
            ])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_scaler = scaler_name
                best_params = grid_search.best_params_

        # Train best model with best scaler on full training data
        best_scaler_instance = scalers[best_scaler]
        pipeline = Pipeline([
            ('scaler', best_scaler_instance),
            ('model', model)
        ])

        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        results.append({
            'Model': model_name,
            'Best Scaler': best_scaler,
            'Best Parameters': best_params,
            'Train Accuracy': accuracy_score(y_train, y_train_pred),
            'Test Accuracy': accuracy_score(y_test, y_test_pred),
            'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
            'Test F1': f1_score(y_test, y_test_pred, average='weighted')
        })

    return results


def main():
    # File paths to datasets
    file_paths = {
        "sp500_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/S&P 500 Futures 2016 24.csv",
        "crudeoil_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/Commodities Crude Oil WTI Futures 2016 24.csv",
        "gold_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/Commodities Gold Futures 2016 24.csv",
        "eur_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/Forex EUR_USD 2016 24.csv",
        "gbp_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/Forex GBP_USD 2016 24.csv",
        "cny_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/Forex USD_CNY 2016 24.csv",
        "jpy_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/Forex USD_JPY 2016 24.csv",
        "usidx_df": "C:/Users/Singh/OneDrive/Desktop/AML/Data/US Dollar Index 2016 24.csv",
    }

    # Load and preprocess datasets
    dfs = load_and_preprocess_data(file_paths)

    # Merge datasets on Date index
    combined_df = merge_datasets(dfs)

    # Clean numeric columns
    clean_numeric_columns(combined_df)

    # Perform exploratory data analysis
    perform_eda(combined_df)

    # Feature engineering
    feature_engineering(combined_df)

    # Create binary target variable
    create_binary_target(combined_df)

    # Drop unnecessary columns
    drop_unnecessary_columns(combined_df)

    # Drop rows with any NaN values after feature engineering
    combined_df.dropna(inplace=True)

    # Prepare features and target for modeling
    X = combined_df.drop(['Binary Movement'], axis=1)
    y = combined_df['Binary Movement']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and get results
    results = train_models(X_train, y_train, X_test, y_test)

    # Print model performance summary
    for result in results:
        print(f"Model: {result['Model']}")
        print(f"Best Scaler: {result['Best Scaler']}")
        print(f"Best Parameters: {result['Best Parameters']}")
        print(f"Train Accuracy: {result['Train Accuracy']:.4f}")
        print(f"Test Accuracy: {result['Test Accuracy']:.4f}")
        print(f"Train F1: {result['Train F1']:.4f}")
        print(f"Test F1: {result['Test F1']:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
