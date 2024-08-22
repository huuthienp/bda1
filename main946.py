#!/usr/bin/env python
# coding: utf-8

# import os
# import sys
# import subprocess
# import pickle
# import zipfile as zp

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# import optuna

from robbie import DataReader, DataPreProcesser
from kye import prepare_data_dbscan, DBSCANExperiment
from dream import DreamExplorer, perform_pca, KMeansGaussianExperiment
from mikael import add_category_information, ClassificationExperiment, ClassificationEvaluator

def perform_kmeans_gaussian(choice, reduced_data):
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
    
def explore_target():
    # Sort the DataFrame so that 'True' values for TARGET come last
    sns.scatterplot(data=df_pp.sort_values(by='target'), \
                    x='current_price', \
                    y='likes_count', \
                    size='discount', \
                    hue='target')
    num_target = df_pp['target'].sum()
    plt.title(f"Best {num_target} Products Scatterplot")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

def perform_classify_experiment(choice, params):
    cexp = ClassificationExperiment(choice, params)
    cexp.split_and_fit(df_pp)
    print(cexp.model)
    y_pred = cexp.model.predict(cexp.X_test)

    ceva = ClassificationEvaluator(cexp.model, cexp.y_test, y_pred)
    ceva.display_report()
    top_prod_proba_per_cat = ceva.get_top_produ_per_cat(df_cat_proba, cat_col, CATEGORIES, COUNT)
    best_cat_df = ceva.get_cat_ratios(df_cat_proba, cat_col, CATEGORIES)
    ceva.plot_confusion_matrix()
    return top_prod_proba_per_cat, best_cat_df

print()
print('Reading data')
zip_file_dir = "A1_2024_Released.zip"
csv_dir = "A1_2024_Unzip"
data_reader = DataReader(zip_file_dir, csv_dir)
combined_df = data_reader.combined_df
dpp = DataPreProcesser(combined_df)
df_pp = dpp.df

CATEGORIES = ['shoes', 'women', 'house', 'men', 'accessories', 'bags', 'jewelry', 'kids', 'beauty']
COUNT = 10
df_cat = add_category_information(df_pp, combined_df)
cat_col = df_cat['category']
df_cat_proba = df_cat.drop(columns='target')

explore_target()

print()
print('Performing clustering with DBSCAN')
# Preprocess the data
df_db, categories_filtered, category_names = prepare_data_dbscan(combined_df, df_pp)
print(df_db)

# Perform clustering and plot
dsce = DBSCANExperiment(df_db, categories_filtered, category_names)
dsce.perform_clustering()
print(df_db)
dsce.plot_clusters()  # Set custom_view to True for specific view angle
dsce.plot_clusters(custom_view=True)  # Set custom_view to True for specific view angle
dsce.plot_detailed_clusters()  # Plot with detailed category markers

print()
print('Performing clustering with KMeans/Gaussian')
dreamer = DreamExplorer(df_pp)
dreamer.plot_box()
reduced_data = perform_pca(dreamer.deal_with_nan())

# Model Selection
while True:
    print()
    print('kmeans, gaussian, or skip?')
    choice = input('The model you want to run: ').lower()
    if 'kmean' in choice or 'gaussian' in choice:
        perform_kmeans_gaussian(choice, reduced_data)
        if input('Repeat (y)? ').lower() == 'y': continue
        else: break
    elif 'skip' in choice: break
    else: continue

print()
print('Performing classification with Random Forest')
params_rf = {'max_depth': 6,
             'criterion': 'log_loss',
             'min_samples_split': 2,
             'max_features': 'log2',
             'max_samples': 0.7264084469432243,
             'n_estimators': 500}

rf_top_prod_proba_per_cat, rf_best_cat_df = perform_classify_experiment(1, params_rf)

print()
print('Performing classification with Logistic Regression')
params_lr = {
    'solver': 'saga',
    'penalty': 'elasticnet',
    'l1_ratio': 0.05,
    'C': 1.25,
    'max_iter': 1000
} # the only needed hyper param is max_iter

lr_top_prod_proba_per_cat, lr_best_cat_df = perform_classify_experiment(2, params_lr)