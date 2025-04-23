import pandas as pd
import numpy as np

def create_features(df):
    """Create additional features for fraud detection"""
    
    # Time-based features
    df['hour_of_day'] = (df['scaled_time'] % 24).astype(int)
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
    
    # Rolling window features
    window_size = 1000
    for col in ['V1', 'V2', 'V3', 'V4', 'V5']:
        df[f'rolling_mean_{col}'] = df[col].rolling(window=window_size).mean()
        df[f'rolling_std_{col}'] = df[col].rolling(window=window_size).std()
    
    # Interaction features
    df['v1_v2_interaction'] = df['V1'] * df['V2']
    df['v3_v4_interaction'] = df['V3'] * df['V4']
    
    # Drop rows with NaN from rolling features
    df = df.dropna()
    
    return df