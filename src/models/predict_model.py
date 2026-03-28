from sklearn.metrics import silhouette_score

# Function to predict and evaluate
def evaluate_model(model, df):
    
    try:
        # Predict the cluster labels on the dataset
        cluster_labels = model.predict(df[['Annual_Income', 'Spending_Score']])

        # Calculate the silhouette score
        score = silhouette_score(df[['Annual_Income', 'Spending_Score']], cluster_labels)

        return cluster_labels, score

    except Exception as e:
        print("Error while predicting and evaluating model:", e)