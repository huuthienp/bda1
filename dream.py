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
    # import plotly
except ImportError:
    print("Some packages are required to be installed")
    print("Installing expected packages")
    install('matplotlib')
    install('numpy')
    install('optuna')
    install('seaborn')
    install('scikit-learn')
    # install('pip')
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
 
class ClusteringAnalysis:
    def __init__(self, data):
        self.data = data

    def plot_elbow_method(self, max_clusters=10):
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)
        plt.plot(range(1, max_clusters + 1), inertia, marker='o')
        plt.title('Elbow Method for cleaned data')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

    def plot_silhouette_analysis(self, min_clusters=2, max_clusters=10):
        silhouette_scores = []
        for k in range(min_clusters, max_clusters):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.data)
            silhouette_scores.append(silhouette_score(self.data, labels))
        plt.plot(range(min_clusters, max_clusters), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        print(f'The highest Silhouette score is {max(silhouette_scores)}')

    def kmean_clustering(self, params):
        kmeans = KMeans().set_params(**params)
        kmeans.fit(self.data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        plt.title('2D PCA of Data with KMeans Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

    def gaussian_model_clustering(self, params):
        gaussian_model = GaussianMixture().set_params(**params)
        gaussian_model.fit(self.data)
        labels = gaussian_model.predict(self.data)
        gaussian_clusters = unique(labels)
        for cluster in gaussian_clusters:
            index = where(labels == cluster)
            plt.scatter(self.data[index, 0], self.data[index, 1])
        plt.title('Gaussian Model Clustering')
        plt.show()

    def plot_gaussian_silhouette_analysis(self, min_clusters=2, max_clusters=7):
        silhouette_scores = []
        for k in range(min_clusters, max_clusters):
            gaussian = GaussianMixture(n_components=k, init_params='kmeans')
            labels = gaussian.fit_predict(self.data)
            silhouette_scores.append(silhouette_score(self.data, labels))
        plt.plot(range(min_clusters, max_clusters), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis for Gaussian Mixture Model')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        print(f'The highest Silhouette score of Gaussian Mixture Model is {max(silhouette_scores)}')
        print(f'The lowest BIC of Gaussian Mixture Model is {min(silhouette_scores)}')

    def plot_bic_aic(self, max_components=20):
        n_components = np.arange(1, max_components + 1)
        models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(self.data) for n in n_components]
        plt.plot(n_components, [model.bic(self.data) for model in models], label='BIC')
        plt.plot(n_components, [model.aic(self.data) for model in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()

    def select_clustering_model(self, choice, params):
        choice = choice.lower()
        if choice == 'kmean':
            self.kmean_clustering(params)
        elif choice == 'gaussian mixture':
            self.gaussian_model_clustering(params)
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
        self.choice = choice.lower()

    def objective(self, trial):
        try:
            if self.choice == 'kmean':
                kmeans = KMeans()
                kmeans_params = {
                    "n_clusters": trial.suggest_int(name="n_clusters", low=2, high=11),
                    "random_state": trial.suggest_int(name="random_state", low=10, high=100)
                }
                kmeans.set_params(**kmeans_params)
                kmeans.fit(self.data)
                kmeans_score = silhouette_score(self.data, kmeans.labels_)
                return kmeans_score

            elif self.choice == 'gaussian mixture':
                gaussian = GaussianMixture(covariance_type="full")
                gaussian_params = {
                    "n_components": trial.suggest_int(name="n_components", low=2, high=11),
                    "random_state": trial.suggest_int(name="random_state", low=10, high=100)
                }
                gaussian.set_params(**gaussian_params)
                gaussian.fit(self.data)
                gmm_score = silhouette_score(self.data, gaussian.predict(self.data))
                return gmm_score

        except ImportError as e:
            print(f"An error occurred: {e}")
            return float('inf')

    def run_study(self, n_trials=5):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study

    @staticmethod
    def visualize_study(study):
        # Plot the optimization history
        optuna.visualization.plot_optimization_history(study).show()
        # Plot the slice plot to see parameter interactions
        optuna.visualization.plot_slice(study).show()
        # Plot the parallel coordinate plot for more insights
        optuna.visualization.plot_parallel_coordinate(study).show()
        #Plot hyperparameter importance
        optuna.visualization.plot_param_importances(study).show()