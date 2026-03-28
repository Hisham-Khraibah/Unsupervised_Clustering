from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_features
from src.models.train_model import train_Kmodel
from src.models.predict_model import evaluate_model
from src.visualization.visualize import (
    plot_pairplot,
    plot_cluster_scatter,
    plot_elbow_method,
    plot_silhouette_method
)

if __name__ == "__main__":
    
    try:
        # Load and preprocess the data
        data_path = "data/raw/mall_customers.csv"
        df = load_and_preprocess_data(data_path)

        # Create features
        df = create_features(df)

        # Plot visuals
        plot_pairplot(df)
        plot_cluster_scatter(df)
        plot_elbow_method(df)
        plot_silhouette_method(df)

        # Train the final KMeans clustering model
        model = train_Kmodel(df)

        # Evaluate the model
        cluster_labels, score = evaluate_model(model, df)
        print(f"Silhouette Score: {score}")

    except Exception as e:
        print("Error while running the main pipeline:", e)