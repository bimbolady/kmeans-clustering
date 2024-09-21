import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os
from kmeans_clustering import load_rdf, is_float, extract_data, preprocess_data, scale_data, kmeans_clustering, \
    apply_pca, evaluate_clustering


class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        """Set up necessary variables for the test"""
        self.file_path = "ss14hak.ttl"
        self.predicates_of_interest = [
            'http://www.census.gov/acs/pums/schema#hasWeightValue',
            'http://www.census.gov/acs/pums/schema#hasReplicateValue',
            'http://www.census.gov/acs/pums/schema#hasIncomeAdjustmentFactor'
        ]
        self.g = load_rdf(self.file_path)
        self.df = extract_data(self.g, self.predicates_of_interest)
        self.df_pivot = preprocess_data(self.df)

    def test_is_float(self):
        """Test if the is_float function works correctly"""
        self.assertTrue(is_float('3.14'))
        self.assertFalse(is_float('abc'))
        self.assertFalse(is_float('None'))

    def test_data_extraction(self):
        """Test if data extraction produces valid non-empty DataFrame"""
        self.assertFalse(self.df.empty, "Data extraction failed; DataFrame is empty.")
        self.assertEqual(len(self.df.columns), 3, "The DataFrame should have 3 columns: subject, predicate, value.")

    def test_preprocess_data(self):
        """Test if data preprocessing correctly handles the pivot and missing values"""
        self.assertFalse(self.df_pivot.empty, "Data preprocessing failed; pivoted DataFrame is empty.")
        self.assertIn('subject', self.df_pivot.columns, "'subject' column should be present after preprocessing.")
        self.assertGreater(len(self.df_pivot.columns), 1, "The pivoted DataFrame should have more than 1 column.")

    def test_normalization(self):
        """Test if the StandardScaler is scaling the data correctly"""
        X_scaled = scale_data(self.df_pivot)

        # Assert the scaled data has the same shape as original data (except 'subject')
        self.assertEqual(X_scaled.shape[1], len(self.df_pivot.columns) - 1,
                         "Scaled data shape should match the original data shape (excluding 'subject').")
        self.assertAlmostEqual(X_scaled.mean(), 0, delta=1e-7, msg="Mean of scaled data should be approximately 0.")
        self.assertAlmostEqual(X_scaled.std(), 1, delta=1e-7,
                               msg="Standard deviation of scaled data should be approximately 1.")

    def test_kmeans_clustering(self):
        """Test if K-Means clustering is working properly"""
        X_scaled = scale_data(self.df_pivot)

        # Apply K-Means clustering with 3 clusters
        kmeans = kmeans_clustering(X_scaled, n_clusters=3)

        self.assertEqual(kmeans.n_clusters, 3, "K-Means should create exactly 3 clusters.")
        self.assertGreater(len(kmeans.labels_), 0, "K-Means should assign labels to the data points.")

    def test_pca_application(self):
        """Test if PCA is applied correctly when more than one feature is available"""
        X_scaled = scale_data(self.df_pivot)

        if X_scaled.shape[1] > 1:
            X_pca = apply_pca(X_scaled)

            self.assertEqual(X_pca.shape[1], 2, "PCA should reduce the data to 2 components.")
        else:
            self.skipTest("Not enough features for PCA.")

    def test_evaluation_metrics(self):
        """Test the evaluation metrics (Silhouette Score and Davies-Bouldin Index)"""
        X_scaled = scale_data(self.df_pivot)

        # Apply K-Means clustering with 3 clusters
        kmeans = kmeans_clustering(X_scaled, n_clusters=3)

        # Calculate silhouette score and Davies-Bouldin index
        silhouette_avg, db_score = evaluate_clustering(X_scaled, kmeans)

        self.assertGreaterEqual(silhouette_avg, 0, "Silhouette score should be >= 0.")
        self.assertGreaterEqual(db_score, 0, "Davies-Bouldin index should be >= 0.")

    def test_model_save(self):
        """Test if the K-Means model is being saved correctly"""
        X_scaled = scale_data(self.df_pivot)

        # Apply K-Means clustering with 3 clusters
        kmeans = kmeans_clustering(X_scaled, n_clusters=3)

        # Save the model
        joblib.dump(kmeans, 'test_kmeans_model.pkl')

        # Check if the model file exists
        self.assertTrue(os.path.exists('test_kmeans_model.pkl'),
                        "The K-Means model should be saved as 'test_kmeans_model.pkl'.")


if __name__ == '__main__':
    unittest.main()
