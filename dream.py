# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import optuna

import matplotlib.pyplot as plt
import pandas as pd

class DreamExplorer:
    def __init__(self, df):
        self.df = df

    def plot_box(self):
        plt.boxplot(self.df)
        plt.xticks(ticks=range(1, len(self.df.columns) + 1), labels=self.df.columns.tolist())
        plt.title('Box Plot to Identify Outliers')
        plt.show()

    def deal_with_nan(self):
        data_clean = self.df.fillna(self.df.median())
        return data_clean

def perform_pca(data):
    # Perform PCA to reduce the dataset to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Print explained variance ratio and PCA components
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    print(f'Components of PCA: {pca.components_}')

    # Create a DataFrame to compare original dataset with PCA components
    # Assuming data is a DataFrame; if it's a NumPy array, adjust accordingly
    if isinstance(data, pd.DataFrame):
        pca_df = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=data.columns)
        print('Original dataset in comparison with PCA components')
        print(pca_df)
    else:
        print("Data is not a DataFrame; PCA components cannot be indexed by column names.")
    return reduced_data

class KMeansGaussianExperiment:
    def __init__(self, data):
        self.data = data

    def k_mean(self, params):
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method for cleaned data')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

        silhouette_scores = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.data)
            silhouette_scores.append(silhouette_score(self.data, labels))

        plt.plot(range(2, 10), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        best_k = 2 + silhouette_scores.index(max(silhouette_scores))
        print(f'The highest Silhouette score of KMeans is {max(silhouette_scores)} at {best_k} clusters.')

        kmeans = KMeans().set_params(**params)
        kmeans.fit(self.data)
        labels = kmeans.labels_

        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50)
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        plt.title('2D PCA of Data with KMeans Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

        return {
            "best_k": best_k,
            "kmeans_model": kmeans,
        }

    def gaussian_model(self, params):
        gaussian_model = GaussianMixture().set_params(**params)
        gaussian_model.fit(self.data)
        label = gaussian_model.predict(self.data)
        gaussian_clusters = np.unique(label)

        for gaussian_cluster in gaussian_clusters:
            index = np.where(label == gaussian_cluster)
            plt.scatter(self.data[index, 0], self.data[index, 1])

        plt.title('Gaussian Model Clustering')
        plt.show()

        silhouette_scores = []
        for k in range(2, 7):
            gaussian = GaussianMixture(n_components=k, init_params='kmeans')
            labels = gaussian.fit_predict(self.data)
            silhouette_scores.append(silhouette_score(self.data, labels))

        plt.plot(range(2, 7), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis for Gaussian Mixture Model')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        best_k = 2 + silhouette_scores.index(max(silhouette_scores))
        print(f'The highest Silhouette score of Gaussian Mixture Model is {max(silhouette_scores)} at {best_k} components.')

        bic_values = []
        component_range = np.arange(1, 11, 1)
        for n in component_range:
            gmm = GaussianMixture(n_components=n)
            gmm.fit(self.data)
            bic_values.append(gmm.bic(self.data))
 
        print(f'The lowest BIC of Gaussian Mixture Model is {min(bic_values)}')
        plt.plot(component_range, bic_values, label='BIC')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC Value')
        plt.title('BIC for Gaussian Mixture Model')
        plt.legend()
        plt.show()

        return {
            "best_k": best_k,
            "gaussian_model": gaussian_model,
            "labels": labels,
            "bic_values": bic_values
        }

    def select_clustering_model(self, choice, params):
        if 'kmean' in choice:
            return self.k_mean(params)
        elif 'gaussian' in choice:
            return self.gaussian_model(params)
        else:
            print("Please type the name of the model you're interested in correctly as above")

    def tune_hyperparameter(self, choice, n_trials=25):
        tuner = HyperparameterTuner(self.data, choice)
        study = tuner.run_study(n_trials)
        return tuner, study

class HyperparameterTuner:
    def __init__(self, data, choice):
        self.data = data
        self.choice = choice

    def objective(self, trial):
        try:
            if 'kmean' in self.choice:
                kmeans = KMeans()
                kmeans_params = {
                    "n_clusters": trial.suggest_int(name="n_clusters", low=2, high=11),
                    "random_state": trial.suggest_int(name="random_state", low=10, high=100)
                }
                kmeans.set_params(**kmeans_params)
                kmeans.fit(self.data)
                kmeans_score = silhouette_score(self.data, kmeans.labels_)
                return kmeans_score

            elif 'gaussian' in self.choice:
                gaussian = GaussianMixture(covariance_type="full")
                gaussian_params = {
                    "n_components": trial.suggest_int(name="n_components", low=2, high=11),
                    "random_state": trial.suggest_int(name="random_state", low=10, high=100)
                }
                gaussian.set_params(**gaussian_params)
                gaussian.fit(self.data)
                gmm_score = silhouette_score(self.data, gaussian.predict(self.data))
                return gmm_score

        except Exception as e:
            print(f"An error occurred: {e}")
            return float('inf')

    def run_study(self, n_trials=25):
        if 'kmean' in self.choice:
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=n_trials)
        elif 'gaussian' in self.choice:
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=n_trials)
        return study

    def visualize_study(self, study):
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_slice(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
        optuna.visualization.plot_param_importances(study).show()

if __name__ == '__main__':
    # Example usage
    from robbie import DataReader, DataPreProcesser
    # define path variables
    zip_file_dir = "A1_2024_Released.zip"
    csv_dir = "A1_2024_Unzip"

    # obtain the data
    data_reader = DataReader(zip_file_dir, csv_dir)
    combined_df = data_reader.combined_df
    dpp = DataPreProcesser(combined_df)
    df_pp = dpp.df

    dreamer = DreamExplorer(df_pp)
    dreamer.plot_box()
    reduced_data = perform_pca(dreamer.deal_with_nan())

    # Model Selection
    print('There are KMean, Gaussian Mixture available')
    choice = input('Which clustering model you want to try? ').lower()

    # Create an instance of the KMeansGaussianExperiment class
    kge = KMeansGaussianExperiment(reduced_data)

    tuner, study_silh = kge.tune_hyperparameter(choice, n_trials=25) # 25-30 minutes
    print(f"Best parameters: {study_silh.best_trial.params}")
    print(f"Best value: {study_silh.best_trial.value}")
    tuner.visualize_study(study)

    # Select and run the chosen model
    observed = kge.perform_clustering(choice, study_silh.best_trial.params)
    print("Observation:")
    print(observed)

    print('Suggestion:')
    if 'kmean' in choice:
        suggested = kge.perform_clustering(choice, { "n_clusters": 3, "random_state": 93 })
    elif 'gaussian' in choice:
        suggested = kge.perform_clustering(choice, { "n_components": 2, "random_state": 49 })
    print(suggested)