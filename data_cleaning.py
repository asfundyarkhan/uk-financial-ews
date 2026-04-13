import pandas as pd
import numpy as np

def load_and_clean_data(ftse_path='ftse.csv', gold_path='gold.csv', silver_path='silver.csv'):
    """
    Load and clean FTSE, Gold, and Silver data by matching common dates.
    
    Parameters:
    -----------
    ftse_path : str
        Path to FTSE CSV file
    gold_path : str
        Path to Gold CSV file
    silver_path : str
        Path to Silver CSV file
        
    Returns:
    --------
    ftse_clean : pd.DataFrame
        Cleaned FTSE data with common dates
    gold_clean : pd.DataFrame
        Cleaned Gold data with common dates
    silver_clean : pd.DataFrame
        Cleaned Silver data with common dates
    """
    
    # Load data, skipping the first 2 rows (Price and Ticker rows), using row 3 as header
    print("Loading data...")
    ftse = pd.read_csv(ftse_path, skiprows=2)
    gold = pd.read_csv(gold_path, skiprows=2)
    silver = pd.read_csv(silver_path, skiprows=2)
    
    # Rename 'Date' column (it's already named Date from the header)
    # Column names should be: Date, Close, High, Low, Open, Volume (but with empty values)
    # Actually the columns are Price, Close, High, Low, Open, Volume
    # Let's properly set column names
    ftse.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    gold.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    silver.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Convert Date column to datetime
    print("Converting dates...")
    ftse['Date'] = pd.to_datetime(ftse['Date'])
    gold['Date'] = pd.to_datetime(gold['Date'])
    silver['Date'] = pd.to_datetime(silver['Date'])
    
    # Remove duplicates based on Date
    print("Removing duplicates...")
    ftse = ftse.drop_duplicates(subset='Date', keep='first')
    gold = gold.drop_duplicates(subset='Date', keep='first')
    silver = silver.drop_duplicates(subset='Date', keep='first')
    
    # Find common dates across all three datasets
    print("Finding common dates...")
    common_dates = set(ftse['Date']) & set(gold['Date']) & set(silver['Date'])
    common_dates = sorted(list(common_dates))
    
    print(f"\nData Summary:")
    print(f"FTSE records: {len(ftse)}")
    print(f"Gold records: {len(gold)}")
    print(f"Silver records: {len(silver)}")
    print(f"Common dates: {len(common_dates)}")
    print(f"Date range: {common_dates[0]} to {common_dates[-1]}")
    
    # Filter each dataset to only include common dates
    print("\nFiltering to common dates...")
    ftse_clean = ftse[ftse['Date'].isin(common_dates)].copy()
    gold_clean = gold[gold['Date'].isin(common_dates)].copy()
    silver_clean = silver[silver['Date'].isin(common_dates)].copy()
    
    # Sort by date
    ftse_clean = ftse_clean.sort_values('Date').reset_index(drop=True)
    gold_clean = gold_clean.sort_values('Date').reset_index(drop=True)
    silver_clean = silver_clean.sort_values('Date').reset_index(drop=True)
    
    # Remove rows with missing values in critical columns
    print("Removing missing values...")
    initial_count = len(ftse_clean)
    ftse_clean = ftse_clean.dropna(subset=['Close', 'Open', 'High', 'Low'])
    gold_clean = gold_clean[gold_clean['Date'].isin(ftse_clean['Date'])]
    silver_clean = silver_clean[silver_clean['Date'].isin(ftse_clean['Date'])]
    
    gold_clean = gold_clean.dropna(subset=['Close', 'Open', 'High', 'Low'])
    ftse_clean = ftse_clean[ftse_clean['Date'].isin(gold_clean['Date'])]
    silver_clean = silver_clean[silver_clean['Date'].isin(gold_clean['Date'])]
    
    silver_clean = silver_clean.dropna(subset=['Close', 'Open', 'High', 'Low'])
    ftse_clean = ftse_clean[ftse_clean['Date'].isin(silver_clean['Date'])]
    gold_clean = gold_clean[gold_clean['Date'].isin(silver_clean['Date'])]
    
    print(f"Removed {initial_count - len(ftse_clean)} rows with missing values")
    print(f"Final record count: {len(ftse_clean)}")
    
    return ftse_clean, gold_clean, silver_clean


def save_cleaned_data(ftse_clean, gold_clean, silver_clean):
    """
    Save cleaned datasets to CSV files.
    
    Parameters:
    -----------
    ftse_clean : pd.DataFrame
        Cleaned FTSE data
    gold_clean : pd.DataFrame
        Cleaned Gold data
    silver_clean : pd.DataFrame
        Cleaned Silver data
    """
    print("\nSaving cleaned data...")
    ftse_clean.to_csv('ftse_cleaned.csv', index=False)
    gold_clean.to_csv('gold_cleaned.csv', index=False)
    silver_clean.to_csv('silver_cleaned.csv', index=False)
    print("Cleaned data saved successfully!")
    print("  - ftse_cleaned.csv")
    print("  - gold_cleaned.csv")
    print("  - silver_cleaned.csv")


if __name__ == "__main__":
    # Load and clean data
    ftse_clean, gold_clean, silver_clean = load_and_clean_data()
    
    # Save cleaned data
    save_cleaned_data(ftse_clean, gold_clean, silver_clean)
    
    # Display first few rows
    print("\n" + "="*50)
    print("Sample of cleaned data:")
    print("="*50)
    print("\nFTSE (first 5 rows):")
    print(ftse_clean.head())
    print("\nGold (first 5 rows):")
    print(gold_clean.head())
    print("\nSilver (first 5 rows):")
    print(silver_clean.head())
