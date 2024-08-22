# -*- coding: utf-8 -*-
"""
# **Section 2 - Program data preprocess (Task 1)**
* Upload the Zip file and the below function should return all of the .CSV files in a single dataframe

## **Imports**
"""
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
try:
    import sys
    import os
    import subprocess
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import zipfile as zp
    import sklearn
    import optuna
    # import plotly.express as px
except ImportError:
    print("Some packages are required to be installed")
    print("Installing expected packages")
    install('matplotlib')
    install('numpy')
    install('optuna')
    install('seaborn')
    install('scikit-learn')
    install('pip')
    # install('plotly')
"""Source code:

1.   https://pandas.pydata.org/
2.   https://www.geeksforgeeks.org/
3.   https://builtin.com/articles/seaborn-pairplot
4.   https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/
5.   https://github.com/christianversloot/machine-learning-articles/blob/main/performing-optics-clustering-with-python-and-scikit-learn.md
6.   https://www.kaggle.com/code/bextuychiev/no-bs-guide-to-hyperparameter-tuning-with-optuna
7.   https://github.com/optuna/optuna
8.   https://optuna.org/
9.   https://medium.com/geekculture/principal-component-analysis-pca-in-feature-engineering-472afa39c27d
10.  https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

"""

"""From visualisation, potentially outliers should be eliminated """
class DataVisualisation:
  def __init__(self, df):
    self.df = df
    
  def print_box_plot(self):
    #identify the outliers
    plt.boxplot(self.df)
    plt.xticks(label=self.df.columns.tolist())
    plt.title('Box Plot to see outliers')
    plt.show()
 
class ClusteringModelsExperiment:
    def __init__(self, data):
        self.data = data

    def k_mean(self, params):
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.reduced_data)
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertia, marker='o')
        plt.title(f'Elbow Method for cleaned data')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

        silhouette_scores = []
        for k in range(2, 10):  
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.reduced_data)
            silhouette_scores.append(silhouette_score(self.reduced_data, labels))

        plt.plot(range(2, 10), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        print(f'The highest Silhouette score of KMeans is {max(silhouette_scores)}')

        kmeans = KMeans().set_params(**params)
        kmeans.fit(self.reduced_data)
        labels = kmeans.labels_

        plt.scatter(self.reduced_data[:, 0], self.reduced_data[:, 1], c=labels, cmap='viridis', s=50)
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        plt.title('2D PCA of Data with KMeans Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

    def gaussian_model(self, params):
        gaussian_model = GaussianMixture().set_params(**params)
        gaussian_model.fit(self.reduced_data)
        label = gaussian_model.predict(self.reduced_data)
        gaussian_clusters = np.unique(label)

        for gaussian_cluster in gaussian_clusters:
            index = np.where(label == gaussian_cluster)
            plt.scatter(self.reduced_data[index, 0], self.reduced_data[index, 1])

        plt.title('Gaussian Model Clustering')
        plt.show()

        silhouette_scores = []
        for k in range(2, 7):  
            gaussian = GaussianMixture(n_components=k, init_params='kmeans')
            labels = gaussian.fit_predict(self.reduced_data)
            silhouette_scores.append(silhouette_score(self.reduced_data, labels))

        plt.plot(range(2, 7), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis for Gaussian Mixture Model')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        print(f'The highest Silhouette score of Gaussian Mixture Model is {max(silhouette_scores)}')

        bic_values = []
        component_range = np.arange(1, 11, 1)
        for n in component_range:
            gmm = GaussianMixture(n_components=n)
            gmm.fit(self.reduced_data)
            bic_values.append(gmm.bic(self.reduced_data))
 
        print(f'The lowest BIC of Gaussian Mixture Model is {min(silhouette_scores)}')
        plt.plot(component_range, bic_values, label='BIC')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC Value')
        plt.title('BIC for Gaussian Mixture Model')
        plt.legend()
        plt.show()

    def select_clustering_model(self, choice, params):
        if 'kmean' in choice:
            return self.k_mean(params)
        elif 'gaussian' in choice:
            return self.gaussian_model(params)
        else:
            print("Please type the name of the model you're interested in correctly as above")

"""
Hyperparameter Tuning 
"""
#Define optimised parameters for each clustering methods
import optuna
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class HyperparameterTuning:
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
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=n_trials)
        return study

    def visualize_study(self, study):
        # Plot the optimization history
        optuna.visualization.plot_optimization_history(study).show()
        # Plot the slice plot to see parameter interactions
        optuna.visualization.plot_slice(study).show()
        # Plot the parallel coordinate plot for more insights
        optuna.visualization.plot_parallel_coordinate(study).show()
        #Plot hyperparameter importance
        optuna.visualization.plot_param_importances(study).show()
