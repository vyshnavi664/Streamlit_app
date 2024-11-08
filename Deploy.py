import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score
# Suppress FutureWarnings from the sklearn.cluster module
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster")



def load_data():
    data = pd.read_csv("World_development_mesurement (1).csv")
    return data
def clean_data(data): 
    # Drop unnecessary columns
    data.drop(['Ease of Business', 'Business Tax Rate'], axis=1, inplace=True)

    # Clean and convert numerical columns
    currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
    for col in currency_cols:
        data[col] = data[col].astype(str).str.replace('[^0-9.]', '', regex=True)
        data[col] = data[col].str.strip()
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col].fillna(data[col].mean(), inplace=True)

    # Impute missing values in numerical and categorical columns
    numerical_cols = ['Birth Rate', 'CO2 Emissions','Days to Start Business','Energy Usage','Health Exp % GDP','Hours to do Tax','Infant Mortality Rate','Internet Usage','Lending Interest','Life Expectancy Female','Life Expectancy Male','Mobile Phone Usage','Number of Records','Population 0-14','Population 15-64','Population 65+','Population Total','Population Urban']
    categorical_cols = ['Country']
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    mode_value = data['Country'].mode()
    data['Country'].fillna(mode_value, inplace=True)

    return data

def handle_outliers(data, column_name, method='trimming'):
    def handle_outliers(data, column_name, method='trimming'):
    # /* Handles outliers in a specified column using the chosen method.

    # *Args:
    # *    data (pd.DataFrame): The DataFrame containing the data.
        # column_name (str): The name of the column to handle outliers in.
    #  *   method (str, optional): The method to use for outlier handling.
            # Options are 'trimming', 'capping', and 'winsorization'.
            # Defaults to 'trimming'.

    # Returns:
        # pd.DataFrame: The DataFrame with outliers handled. */

     if method == 'trimming':
        z_scores = np.abs((data[column_name] - data[column_name].mean()) / data[column_name].std())
        outliers = data[z_scores > 3].index
        data.drop(outliers, inplace=True)
     elif method == 'capping':
        threshold = data[column_name].quantile(0.95)
        data[column_name] = np.where(data[column_name] > threshold, threshold, data[column_name])
     elif method == 'winsorization':
        lower_percentile = 0.05
        upper_percentile = 0.95
        lower_bound = data[column_name].quantile(lower_percentile)
        upper_bound = data[column_name].quantile(upper_percentile)
        data[column_name] = np.clip(data[column_name], lower_bound, upper_bound)

    return data

def preprocess_data(data):
    # Scale numerical columns
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Encode categorical columns
    label_encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:

        data[col] = label_encoder.fit_transform(data[col])


    return data



def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components

from sklearn.cluster import KMeans

def kmeans_clustering(data, n_clusters):
  """
  Performs KMeans clustering on the given data.

  Args:
      data: A numpy array representing the data points.
      n_clusters: The desired number of clusters.

  Returns:
      A numpy array containing the cluster labels for each data point.
  """
  # Initialize the KMeans object
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)

  # Fit the KMeans model to the data
  kmeans.fit(data)

  # Extract the cluster labels
  labels = kmeans.labels_

  # Return the cluster labels
  return labels

def hierarchical_clustering_average(data, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    hc.fit(data)
    labels = hc.labels_
    return labels
# ... Implement other clustering algorithms (DBSCAN, MeanShift, Spectral Clustering, Affinity Propagation) ...

def evaluate_clustering(data, labels):
    silhouette_avg = silhouette_score(data, labels)
    calinski_harabasz_score_value = calinski_harabasz_score(data, labels)
    davies_bouldin_score_value = davies_bouldin_score(data, labels)

    st.write("Silhouette Coefficient:", silhouette_avg)
    st.write("Calinski-Harabasz Index:", calinski_harabasz_score_value)
    st.write("Davies-Bouldin Index:", davies_bouldin_score_value)

def visualize_clusters(data, labels):
    # Create a figure and axis objects
  fig, ax = plt.subplots(figsize=(10, 6))

  # Scatter plot with color-coded labels
  ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

  # Add labels and title
  ax.set_title("Cluster Visualization")
  ax.set_xlabel("Feature 1")
  ax.set_ylabel("Feature 2")

  # Display the plot in Streamlit using st.pyplot
  st.pyplot(fig)

  # Close the figure (optional)
  plt.close(fig)  # This line is optional, but helps with memory management

def main():
    st.title("World Development Analysis")

    # Load and preprocess data
    data = load_data()
    data = clean_data(data)
    data = handle_outliers(data, 'GDP')  # Example outlier handling
    data = preprocess_data(data)

    
    # Apply PCA
    st.subheader("PCA")
    n_components = st.slider("Number of Principal Components", 1, 10, 2)
    principal_components = apply_pca(data, n_components)

    # Clustering
    st.subheader("Clustering")
    clustering_method = st.selectbox("Select Clustering Method", ['K-Means', 'Hierarchical (Average)'])
    if clustering_method == 'K-Means':
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        labels = kmeans_clustering(principal_components, n_clusters)
        evaluate_clustering(principal_components, labels)
        visualize_clusters(principal_components[:, :2], labels)
    elif clustering_method == 'Hierarchical (Average)':
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        labels = hierarchical_clustering_average(principal_components, n_clusters)
        evaluate_clustering(principal_components, labels)
        visualize_clusters(principal_components[:, :2], labels)

if __name__ == "__main__":
    main()