import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load the dataset
data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates={'datetime': [0, 1]},
                   infer_datetime_format=True, na_values='?', low_memory=False, index_col='datetime')

# Fill missing values with median and reduce dataset for example purposes
data = data.fillna(data.median()).resample('D').mean()

# Feature engineering
data['day'] = data.index.day
data['month'] = data.index.month
data['year'] = data.index.year

# Selecting features and target
X = data[['day', 'month', 'year', 'Global_active_power']]
y = data['Global_active_power'].shift(-1)  # Next day's power consumption
X = X[:-1]
y = y.dropna()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = SVR(kernel='rbf')
model.fit(X_train_scaled, y_train)

# Save model and scaler
dump(model, 'models/svr_model.joblib')
dump(scaler, 'models/scaler.joblib')
