import pandas as pd

# create features
def create_features(df):
    
    try:
        # store the processed dataset in data/processed
        df.to_csv('data/processed/Processed_Mall_Customers.csv', index=None)

        return df

    except Exception as e:
        print("Error while creating features:", e)