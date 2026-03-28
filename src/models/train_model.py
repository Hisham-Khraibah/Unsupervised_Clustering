from sklearn.cluster import KMeans
import pickle

# Function to train the model
def train_Kmodel(df):
    
    try:
        # Train the KMeans clustering model using Annual_Income and Spending_Score
        model = KMeans(n_clusters=5, random_state=0).fit(df[['Annual_Income', 'Spending_Score']])

        # Save the trained model
        with open('models/Kmodel.pkl', 'wb') as f:
            pickle.dump(model, f)

        return model

    except Exception as e:
        print("Error while training the model:", e)