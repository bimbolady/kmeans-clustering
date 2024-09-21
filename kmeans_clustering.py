import rdflib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os


# Function to load RDF data from Turtle file
def load_rdf(file_path):
    g = rdflib.Graph()
    g.parse(file_path, format="ttl")
    return g


# Function to check if a string can be converted to float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Function to extract triples based on selected predicates
def extract_data(graph, predicates_of_interest):
    data = []
    for s, p, o in graph:
        if str(p) in predicates_of_interest:
            # Check if the object can be converted to a float
            if is_float(o):
                data.append((str(s), str(p), float(o)))  # Only append numeric values
    return pd.DataFrame(data, columns=['subject', 'predicate', 'value'])


# Function to preprocess and pivot the data
def preprocess_data(df):
    # Pivot the data so each row is a subject, and each predicate is a column
    df_pivot = df.pivot(index='subject', columns='predicate', values='value').fillna(0).reset_index()
    return df_pivot


# Function to scale the data using StandardScaler
def scale_data(df_pivot):
    # Ensure 'subject' is not included in the scaling process
    X = df_pivot.drop(columns=['subject'], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Function to perform K-Means clustering
def kmeans_clustering(X_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_scaled)
    return kmeans


# Function to perform PCA (optional, based on feature size)
def apply_pca(X_scaled):
    if X_scaled.shape[1] > 1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca
    return None


# Function to plot the clustering results
def plot_clusters(X_pca, kmeans):
    if X_pca is not None:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
        plt.title('K-Means Clustering of RDF Data (2D PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()
    else:
        print("Not enough features for PCA. Only one feature found.")


# Function to evaluate the clustering using Silhouette Score and Davies-Bouldin Index
def evaluate_clustering(X_scaled, kmeans):
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    db_score = davies_bouldin_score(X_scaled, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {db_score}")
    return silhouette_avg, db_score


# Function to apply the Elbow Method to find the optimal number of clusters
def elbow_method(X_scaled):
    inertia = []
    for n in range(1, 10):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 10), inertia, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


# Function to save the K-Means model and results to files
def save_model_and_results(kmeans, df_pivot):
    joblib.dump(kmeans, 'kmeans_model.pkl')
    df_pivot.to_csv('clustered_data.csv', index=False)
    print("K-Means model and clustered data have been saved.")


# Main workflow
def main():
    file_path = "ss14hak.ttl"

    # Define the predicates you're interested in
    predicates_of_interest = [
        'http://www.census.gov/acs/pums/schema#hasWeightValue',
        'http://www.census.gov/acs/pums/schema#hasReplicateValue',
        'http://www.census.gov/acs/pums/schema#hasIncomeAdjustmentFactor',
        'http://www.census.gov/acs/pums/schema#hasSerialNumber',
        'http://www.census.gov/acs/pums/schema#hasHousingValue'
    ]

    # Load RDF data
    g = load_rdf(file_path)

    # Extract relevant data based on predicates
    df = extract_data(g, predicates_of_interest)

    # Preprocess data (pivot and fill missing values)
    df_pivot = preprocess_data(df)

    if df_pivot.empty:
        raise ValueError(
            "No data extracted for clustering. Please check the predicates of interest or the dataset format.")

    print(df_pivot.head())  # Check the preprocessed data

    # Scale the data
    X_scaled = scale_data(df_pivot)

    # Apply K-Means clustering
    kmeans = kmeans_clustering(X_scaled, n_clusters=3)

    # Add cluster labels back to the DataFrame
    df_pivot['cluster'] = kmeans.labels_

    print(df_pivot[['subject', 'cluster']].head())  # Display the subject and their assigned cluster

    # Apply PCA (if applicable)
    X_pca = apply_pca(X_scaled)

    # Plot the clustering results
    plot_clusters(X_pca, kmeans)

    # Evaluate the clustering
    evaluate_clustering(X_scaled, kmeans)

    # Use the Elbow method to find the optimal number of clusters
    elbow_method(X_scaled)

    # Save the model and results
    save_model_and_results(kmeans, df_pivot)


if __name__ == "__main__":
    main()
