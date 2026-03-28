import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def plot_pairplot(data):
    """
    Plot a pairplot for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    try:
        sns.pairplot(data[['Age', 'Annual_Income', 'Spending_Score']])
        plt.tight_layout()
        plt.savefig("pairplot.png")
        plt.show()
    
    except Exception as e:
        print("Error while plotting pairplot:", e)

def plot_cluster_scatter(data):
    """
    Plot the customer clusters using Annual Income and Spending Score.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    try:
        kmodel = KMeans(n_clusters=5, random_state=0).fit(data[['Annual_Income', 'Spending_Score']])

        # Put predicted clusters into dataframe
        plot_data = data.copy()
        plot_data['Cluster'] = kmodel.predict(plot_data[['Annual_Income', 'Spending_Score']])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Annual_Income',
            y='Spending_Score',
            data=plot_data,
            hue='Cluster',
            palette='colorblind',
            ax=ax
        )

        # Plot cluster centers
        centers = kmodel.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

        plt.xlabel('Annual_Income')
        plt.ylabel('Spending_Score')
        plt.title('Customer Segments')
        plt.tight_layout()
        fig.savefig("cluster_plot.png")
        plt.show()
    
    except Exception as e:
        print("Error while plotting cluster scatter:", e)

def plot_elbow_method(data):
    """
    Plot elbow method for clusters 3 to 8.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    try:
        k = range(3, 9)
        K = []
        WCSS = []

        for i in k:
            kmodel = KMeans(n_clusters=i, random_state=0).fit(data[['Annual_Income', 'Spending_Score']])
            wcss_score = kmodel.inertia_
            WCSS.append(wcss_score)
            K.append(i)

        wss = pd.DataFrame({'cluster': K, 'WCSS_Score': WCSS})

        fig, ax = plt.subplots(figsize=(8, 6))
        wss.plot(x='cluster', y='WCSS_Score', ax=ax)
        plt.xlabel('No. of clusters')
        plt.ylabel('WSS Score')
        plt.title('Elbow Plot')
        plt.grid(True)
        plt.tight_layout()
        fig.savefig("elbow_plot.png")
        plt.show()
    
    except Exception as e:
        print("Error while plotting elbow method:", e)

def plot_silhouette_method(data):
    """
    Plot silhouette scores for clusters 3 to 8.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    try:
        k = range(3, 9)
        K = []
        ss = []

        for i in k:
            kmodel = KMeans(n_clusters=i, random_state=0).fit(data[['Annual_Income', 'Spending_Score']])
            ypred = kmodel.labels_
            sil_score = silhouette_score(data[['Annual_Income', 'Spending_Score']], ypred)
            K.append(i)
            ss.append(sil_score)

        wss = pd.DataFrame({'cluster': K, 'Silhouette_Score': ss})

        fig, ax = plt.subplots(figsize=(8, 6))
        wss.plot(x='cluster', y='Silhouette_Score', ax=ax)
        plt.xlabel('No. of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Plot')
        plt.tight_layout()
        fig.savefig("silhouette_plot.png")
        plt.show()
    
    except Exception as e:
        print("Error while plotting silhouette method:", e)