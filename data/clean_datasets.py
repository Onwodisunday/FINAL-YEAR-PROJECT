
import pandas as pd
import os

# Define file paths
files = {
    'AWS': 'AWS_Cloud_Dataset.csv',
    'Azure': 'Azure_Cloud_Dataset.csv',
    'GCP': 'GCP_Cloud_Dataset.csv'
}

def clean_dataset(name, filename):
    print(f"--- Cleaning {name} ({filename}) ---")
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    df = pd.read_csv(filename)
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    # 1. Remove Duplicates
    df.drop_duplicates(inplace=True)
    dedup_rows = len(df)
    print(f"Rows after removing duplicates: {dedup_rows} (Removed {initial_rows - dedup_rows})")

    # 2. Handle Missing Values
    # Check for nulls
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"Found {null_counts} missing values. Dropping rows with missing values...")
        df.dropna(inplace=True)
    else:
        print("No missing values found.")
    
    after_na_rows = len(df)
    print(f"Rows after handling missing values: {after_na_rows}")

    # 3. Data Type Verification & Standardization
    # Ensure timestamp is valid datetime
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("Verified 'timestamp' column is valid datetime.")
        except Exception as e:
            print(f"Error converting timestamp: {e}")
    
    # Ensure numeric columns are numeric
    numeric_cols = ['cpu_usage', 'memory_usage', 'net_io', 'disk_io', 'vCPU', 'RAM_GB', 'price_per_hour', 'latency_ms', 'throughput', 'cost', 'utilization']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where numeric conversion failed (became NaN)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"Final rows after data type checks: {final_rows}")

    # Save cleaned data
    # We overwrite the file or save as new? 
    # User said "clean this data", implying modifying them to be correct.
    # I'll overwrite for now as it's a 'clean' operation on the split files.
    df.to_csv(filename, index=False)
    print(f"Saved cleaned data to {filename}\n")

# Run cleaning
for name, filename in files.items():
    clean_dataset(name, filename)

print("All datasets processed.")
