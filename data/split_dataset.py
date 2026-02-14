
import pandas as pd
import os

# Read the dataset
df = pd.read_csv('Cloud_Dataset.csv')

# Get unique cloud providers
providers = df['cloud_provider'].unique()

print(f"Found providers: {providers}")

for provider in providers:
    # Filter data for the provider
    provider_df = df[df['cloud_provider'] == provider]
    
    # Create filename
    filename = f"{provider}_Cloud_Dataset.csv"
    
    # Save to CSV
    provider_df.to_csv(filename, index=False)
    print(f"Saved {len(provider_df)} rows to {filename}")

print("Done!")
