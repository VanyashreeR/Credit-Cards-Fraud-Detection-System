
### 2. data_preprocessing.py

```python
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    """Load and preprocess the raw data"""
    df = pd.read_csv('../data/creditcard.csv')
    
    # Scale time and amount features
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Save processed data
    df.to_csv('../data/processed_data.csv', index=False)
    joblib.dump(scaler, '../models/scaler.pkl')
    
    return df

def split_data(df):
    """Split data into train and test sets"""
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test