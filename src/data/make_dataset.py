import pandas as pd

def load_and_preprocess_data(data_path):
    
    try:
        # Import the data from 'mall_customers.csv'
        df = pd.read_csv(data_path)

        return df

    except Exception as e:
        print("Error while loading and preprocessing data:", e)