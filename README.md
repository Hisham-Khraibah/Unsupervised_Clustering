# mall_customer_clustering
This app has been built using Streamlit and deployed with Streamlit Community Cloud.



This application groups mall customers into different segments based on their characteristics using unsupervised machine learning (KMeans clustering). The goal is to help businesses understand customer behavior and improve marketing strategies.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter customer details such as age, income, and spending score.
- Predicts which cluster a customer belongs to.
- Visualizations including:
  - Correlation heatmap
  - Elbow method for optimal clusters
  - Cluster scatter plot
- Model and scaler saved for reuse.

## Dataset
The application uses the **Mall Customers Dataset**, which includes the following features:
- Customer_ID (removed during preprocessing)
- Gender
- Age
- Annual_Income
- Spending_Score

These features are used to group customers into clusters.

## Machine Learning Approach
- **Algorithm Used:** KMeans Clustering
- **Preprocessing:**
  - Missing value handling
  - One-hot encoding for categorical variables (Gender)
  - Feature scaling using MinMaxScaler
- **Evaluation:**
  - Silhouette Score
  - Elbow Method (WCSS)

## Project Structure
src/
│
├── data/
│ └── make_dataset.py
│
├── features/
│ └── build_features.py
│
├── models/
│ ├── train_model.py
│ └── predict_model.py
│
├── visualization/
│ └── visualize.py
│
main.py
streamlit.py
requirements.txt

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For clustering (KMeans) and evaluation.
- **Pandas** and **NumPy**: For data preprocessing.
- **Matplotlib** and **Seaborn**: For data visualization.

## How to Run the Project
### 1. Install dependencies
`pip install -r requirements.txt`

### 2. Run the pipeline
`python main.py`

### 3. Run the Streamlit app
`python -m streamlit run streamlit.py`

## Output
- Trained model saved as:
  `models/Kmodel.pkl`
- Scaler saved as:
  `models/scaler.pkl`
- Visualizations displayed during execution