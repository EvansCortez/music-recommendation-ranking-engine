import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(sample_size=500000):
    print("🚀 Starting Preprocessing...")

    # 1. Load Data
    # Ensure your CSVs are in the 'data' folder!
    train = pd.read_csv('data/train.csv', nrows=sample_size)
    songs = pd.read_csv('data/songs.csv')
    members = pd.read_csv('data/members.csv')

    print(f"📊 Merging datasets (Sample size: {sample_size})...")
    
    # 2. Merging
    # Left join ensures we keep all training interactions
    df = train.merge(songs, on='song_id', how='left')
    df = df.merge(members, on='msno', how='left')

    # 3. Feature Engineering: Dates
    print("📅 Processing date features...")
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m%d', errors='coerce')
    
    # Create 'account_age' feature (days since registration)
    # Using a fixed date for consistency in the project
    reference_date = pd.to_datetime('2017-02-01') 
    df['account_age'] = (reference_date - df['registration_init_time']).dt.days
    
    # 4. Handle Missing Values
    print("🧹 Cleaning missing values...")
    df['gender'] = df['gender'].fillna('unknown')
    df['song_length'] = df['song_length'].fillna(df['song_length'].mean())
    # Fill categorical NaNs with a string so LabelEncoder works
    cat_cols = ['genre_ids', 'artist_name', 'composer', 'lyricist', 'source_system_tab', 'source_screen_name', 'source_type']
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')

    # 5. Label Encoding
    # This turns "Pop" into 1, "Rock" into 2, etc.
    print("🔢 Encoding categorical variables...")
    le = LabelEncoder()
    cols_to_encode = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 
                      'source_type', 'gender', 'artist_name']
    
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col].astype(str))

    # 6. Save the cleaned data
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
        
    df.to_csv('data/processed/train_cleaned.csv', index=False)
    print("✅ Success! Cleaned data saved to 'data/processed/train_cleaned.csv'")

if __name__ == "__main__":
    preprocess_data()