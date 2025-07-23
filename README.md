# Market Movement Prediction using Financial Indicators

This project uses multiple financial time series datasets (e.g., S&P 500 Futures, Gold, Oil, Forex rates) to predict the daily movement of the S&P 500 index using various classification models. It involves data cleaning, feature engineering, exploratory analysis, and model selection with different scalers and hyperparameters.

---

## 📁 Datasets Used

The following datasets are used (from 2016 to 2024):

- S&P 500 Futures
- Crude Oil WTI Futures
- Gold Futures
- EUR/USD
- GBP/USD
- USD/CNY
- USD/JPY
- US Dollar Index

---

## 📌 Project Workflow

### 1. **Data Loading & Preprocessing**
- Load all datasets
- Handle missing values (forward and backward fill)
- Align datasets on the datetime index

### 2. **Data Cleaning**
- Convert string-based numeric columns (e.g., "1,200.5M", "3.2%") to floats
- Drop or transform non-informative features

### 3. **EDA (Exploratory Data Analysis)**
- Descriptive statistics
- Histograms
- Correlation heatmap
- Time-series plots

### 4. **Feature Engineering**
- Momentum indicators
- Moving averages (SMA 5/10/20)
- Price volatility
- Ratios (Gold/S&P, Dollar Index/S&P)
- RSI (Relative Strength Index)
- Binary movement label: `-1` (down), `0` (no change), `1` (up)

### 5. **Modeling**
Models evaluated with 5-fold cross-validation and multiple scalers:
- **Logistic Regression** (L1/L2 penalty)
- **Linear SVM**
- **Gradient Boosting Classifier**

Each model is evaluated using:
- Accuracy (Train/Test)
- Weighted F1 Score (Train/Test)

---

## 🔍 Results Summary

Each model is trained using the best-performing scaler and hyperparameters. Final performance metrics include:

- Train & Test Accuracy
- Train & Test F1 Score
- Optimal hyperparameters for each model

📌 Results are printed in a structured format at the end of the script.

---

## 🚀 How to Run

1. Place all `.csv` files in the correct path as specified in `file_paths` in the script.
2. Run the Python script:

```bash
python market_movement_prediction.py
```

## 🛠 Requirements

Install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📎 File Structure

```bash
📦 project-root/
│
├── market_movement_prediction.py      # Main script
├── README.md                          # Project overview
├── data/                              # Folder for raw CSV files
│   ├── S&P 500 Futures 2016 24.csv
│   ├── Commodities Gold Futures 2016 24.csv
│   └── ...
```

## 📬 Contact
Shikha Singh
📧 singh.shikha692000@gmail.com
📍 Syracuse, NY
[LinkedIn](https://www.linkedin.com/in/shikha--singh/)


