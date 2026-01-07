import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CRITICAL COLUMN MAPPING ---
COL_FILM = 'Film_Name'
COL_RELEASE = 'Release_Date'
COL_CATEGORY = 'Category'
COL_LANGUAGE = 'Language'
COL_VIEWS = 'Number_of_Views'
COL_VIEW_MONTH_STR = 'Viewing_Month'
DATA_PATH = os.path.join("data", "Film_Dataset.csv")

def load_and_preprocess(path=DATA_PATH):
    """
    Loads data, cleans date formats, creates necessary features, and 
    separates the full dataset from the filtered training set.
    Returns: df_model (for training), df_full (for overall analysis/prediction).
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, None
        
    # Convert dates and create features
    df[COL_RELEASE] = pd.to_datetime(df[COL_RELEASE], errors='coerce')
    # Correctly handle the YYYY-MM date string format
    df[COL_VIEW_MONTH_STR] = pd.to_datetime(df[COL_VIEW_MONTH_STR], format='%Y-%m', errors='coerce')

    df = df.dropna(subset=[COL_RELEASE, COL_VIEW_MONTH_STR, COL_VIEWS]).reset_index(drop=True)

    # Feature Engineering
    df['Release_Year'] = df[COL_RELEASE].dt.year
    df['Viewing_Year'] = df[COL_VIEW_MONTH_STR].dt.year
    df['Movie_Age'] = 2025 - df['Release_Year']
    df['Month_Number'] = df[COL_VIEW_MONTH_STR].dt.month

    # Keep original columns for dashboard display
    df['Language_original'] = df[COL_LANGUAGE].copy()
    df['Category_original'] = df[COL_CATEGORY].copy()
    df_full = df.copy() # The complete, cleaned dataset for prediction/display

    # --- Temporal Filtering (Required by Assignment) ---
    # Training data must be EVERYTHING BEFORE December 2025
    df_train_ready = df[df[COL_VIEW_MONTH_STR] < '2025-12-01'].copy()

    # Create dummy variables for ML model
    df_model = pd.get_dummies(df_train_ready, columns=[COL_CATEGORY, COL_LANGUAGE], drop_first=True)

    # Sort chronologically for time-series split
    df_model = df_model.sort_values(COL_VIEW_MONTH_STR).reset_index(drop=True)

    return df_model, df_full

def get_training_data(df_model):
    """Splits the chronologically sorted data into training and test sets."""
    n_total = len(df_model)
    n_train = int(n_total * 0.8)  # 80/20 time-based split
    
    train_data = df_model.iloc[:n_train]
    test_data = df_model.iloc[n_train:]

    drop_cols = [COL_VIEWS, COL_FILM, COL_VIEW_MONTH_STR, COL_RELEASE, 'Language_original', 'Category_original']
    
    X_train = train_data.drop([c for c in drop_cols if c in train_data.columns], axis=1)
    y_train = train_data[COL_VIEWS]
    X_test = test_data.drop([c for c in drop_cols if c in test_data.columns], axis=1)
    y_test = test_data[COL_VIEWS]
    
    return X_train, y_train, X_test, y_test

def calculate_historical_totals(df_full):
    """Calculates aggregates used for the bar and pie charts."""
    df_categories = df_full.groupby('Category_original')[COL_VIEWS].sum().sort_values(ascending=False).reset_index(name='Total_Views')
    df_languages = df_full.groupby('Language_original')[COL_VIEWS].sum().sort_values(ascending=False).reset_index(name='Total_Views')
    return df_categories, df_languages