import subprocess
import sys

try:
    subprocess.check_call([sys.executable, '-m', 'ensurepip'])
except Exception as e:
    print(e, file=sys.stderr)

def ensure_module_installed(module_name):
    try:
        __import__(module_name)
        print(f'Module "{module_name}" imported successfully.')
    except ImportError:
        print(f'Module "{module_name}" not found, attempting to install...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_name])
            print(f'Module "{module_name}" installed successfully.')
        except Exception as e:
            print(e, file=sys.stderr)

modules = [
    "os",
    "sys",
    "subprocess",
    "zipfile",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "optuna",
]

for m in modules:
    ensure_module_installed(m)
#
import os
import zipfile as zp

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataReader:
  def __init__(self, zip_file_dir, csv_dir):
    self.csv_dir = csv_dir
    self.zip_file_dir = zip_file_dir
    self.combined_df = self.process_all()

  def process_all(self):
    self.unzip()
    return self.combine_csv()

  def unzip(self):
    with zp.ZipFile(self.zip_file_dir, "r") as zip_ref:
      zip_ref.extractall(self.csv_dir)

  def combine_csv(self):
    df_list = []
    wdir = os.path.join(self.csv_dir, "A1_2024_Released")
    list_dir = os.listdir(wdir)
    list_dir_csv = [filename for filename in list_dir if filename.endswith('.csv')]
    for filename in list_dir_csv:
      file_path = os.path.join(wdir, filename)
      df = pd.read_csv(file_path)
      df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

class DataPreProcesser:
  def __init__(self, df):
    self.df = df
    self.preprocess_all()


  def preprocess_all(self):
    self.keep_numeric_cols()
    self.df = self.stdsclar()

  def keep_numeric_cols(self):
    columns_to_keep = [col for col in self.df.columns if self.df[col].dtype in ['float64', 'int64']]
    self.df = self.df[columns_to_keep]
    self.df  = self.df.drop("id", axis=1)

  def stdsclar(self):
    scaler = StandardScaler()
    discount_tmp = scaler.fit_transform(self.df[['discount']])
    likes_tmp = scaler.fit_transform(np.log1p(self.df[['likes_count']]))
    metric = discount_tmp+likes_tmp

    df_sclr = scaler.fit_transform(self.df)  # Ensure only numeric columns are scaled
    df_sclr = pd.DataFrame(df_sclr, columns=self.df.columns)
    df_sclr['metric'] = metric.ravel()
    df_sclr['target'] = np.where(df_sclr['metric']>2, 1, 0)
    df_sclr = df_sclr.drop("metric", axis=1)

    return df_sclr
#
#
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.pyplot as plt

def prepare_data_dbscan(combined_df, df_pp):
    le = LabelEncoder()
    categories_numeric = le.fit_transform(combined_df['category'])
    features = ['current_price', 'raw_price', 'discount', 'likes_count']
    df_db = df_pp[features].copy()

    # Calculate Z-scores and filter outliers
    z_scores = np.abs(stats.zscore(df_db))
    threshold = 3
    non_outliers = (z_scores < threshold).all(axis=1)
    df_db = df_db[non_outliers]
    categories_filtered = categories_numeric[non_outliers]

    return df_db.copy(), categories_filtered, le.classes_

class DBSCANExperiment:
    def __init__(self, df_db, categories_filtered, category_names):
        self.df_db = df_db
        self.categories_filtered = categories_filtered
        self.category_names = category_names
        self.labels = None

    def perform_clustering(self):
        db = DBSCAN(eps=0.5, min_samples=35).fit(self.df_db)
        self.labels = db.labels_
        self.df_db['Cluster'] = self.labels

    def plot_clusters(self, custom_view=False):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.df_db['raw_price'], self.df_db['likes_count'], self.df_db['discount'], c=self.df_db['Cluster'], cmap='rainbow')
        ax.set_xlabel('Raw Price')
        ax.set_ylabel('Likes count')
        ax.set_zlabel('Discount')
        plt.title('3D Plot of DBSCAN Clusters')

        if custom_view:
            ax.view_init(elev=20., azim=30)

        plt.show()

    def plot_detailed_clusters(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a list of markers for different categories
        markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D', '|']

        # Map the numerical categories to markers
        category_markers = np.array([markers[i % len(markers)] for i in self.categories_filtered])

        for cluster in np.unique(self.labels):
            if cluster == -1:  # Skip noise points if needed
                continue

            for i, marker in enumerate(markers):
                cluster_points = (self.labels == cluster) & (category_markers == marker)
                ax.scatter(self.df_db.loc[cluster_points, 'raw_price'],
                           self.df_db.loc[cluster_points, 'likes_count'],
                           self.df_db.loc[cluster_points, 'discount'],
                           label=f'Cluster {cluster}, {self.category_names[i]}',
                           marker=marker)

        ax.set_xlabel('Raw Price')
        ax.set_ylabel('Likes count')
        ax.set_zlabel('Discount')
        ax.set_title('3D Plot of DBSCAN Clusters with Categories')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
        plt.show()

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

    def perform_clustering(self, choice, params):
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
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study

    def visualize_study(self, study):
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_slice(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
        optuna.visualization.plot_param_importances(study).show()

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def add_category_information(df_pp: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    "Raw data is just the dataset with column with category information, need to add category information to preprocessed data"
    # df_pp = pd.read_csv("df_pp.csv", index_col=False)
    # combined_df = pd.read_csv("data.csv", index_col=False)
    df_cat = df_pp.copy()
    df_cat["category"] = combined_df["category"]
    return df_cat

class ClassificationExperiment:
    def __init__(self, model_choice, params):
        match model_choice:
            case 1:
                self.model = RandomForestClassifier(**params, random_state=42)
            case 2:
                self.model = LogisticRegression(**params, random_state=42)
        self.X_test = None
        self.y_test = None

    #TODO: Just simple preprocess, remove when common preprocessing decided
    def split_and_fit(self, df_pp: pd.DataFrame) -> RandomForestClassifier:
        X = df_pp.drop('target', axis=1)
        y = df_pp['target'].values
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
        self.model.fit(X_train, y_train)

class ClassificationEvaluator:
    def __init__(self, model, y_test, y_pred):
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_confusion_matrix(self) -> None:
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {self.model.__class__.__name__}")
        plt.show()

    def display_report(self) -> None:
        print('Classification Report:')
        print(classification_report(self.y_test, self.y_pred))
        recall = recall_score(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average='binary')
        print(f"The recall of {self.model.__class__.__name__} is: {recall:.9f}")
        print(f"The accuracy of {self.model.__class__.__name__} is: {accuracy:.9f}")
        print(f"The f1 score of {self.model.__class__.__name__} is: {f1:.9f}")
        print()

    def get_top_produ_per_cat(self, df_cat, category_column, categories, count) -> dict:
        top_percat_produ = {}
        for cat in categories:
            # Filter data for the current category
            filtered_data = df_cat[category_column == cat]
            # Predict probabilities for the current category
            proba_data = self.model.predict_proba(filtered_data.drop(columns='category'))
            # Convert probabilities to DataFrame
            proba_df = pd.DataFrame(proba_data, columns=['Class_0', 'Class_1'], index=filtered_data.index)
            top_idx = proba_df['Class_1'].nlargest(count).index
            top_percat_produ[cat] = proba_df.loc[top_idx]
        return top_percat_produ

    def get_top_produ(self, df_pp, count):
        proba_data = self.model.predict_proba(df_pp.drop(columns='target'))
        proba_df = pd.DataFrame(proba_data, columns=['Class_0', 'Class_1'], index=df_pp.index)
        top_produ_df = proba_df['Class_1'].nlargest(count)
        return top_produ_df

    def get_cat_ratios(self, df_cat, category_column, categories):
        predictions = {}

        for cat in categories:
            # Filter data for the current category and make predictions
            filtered_data = df_cat[category_column == cat]
            predictions[cat] = self.model.predict(filtered_data.drop(columns='category'))

        # Initialize a list to store the results
        results = []

        # Populate the results list with data
        for cat, preds in predictions.items():
            good = np.sum(preds)
            total = len(preds)
            ratio = good / total if total > 0 else 0
            results.append({
                'Category': cat,
                'Nr of Good Products': good,
                'Total Items': total,
                'Ratio': ratio
            })

        # Create a DataFrame from the results
        df = pd.DataFrame(results)
        print(df)

        # Find and print the best category
        best_category = df.loc[df['Ratio'].idxmax()]['Category']
        print(f"Best category: {best_category}")
        return df

    def visualize_best_category(self, prod_df: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Ratio', data=prod_df, palette='Blues_d')
    
        # Add titles and labels
        plt.xlabel('Category')
        plt.ylabel('Ratio of "Good" Products')
        plt.title(f'Ratio of "Good Products" by Category - {self.model.__class__.__name__}')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def perform_kmeans_gaussian(choice, reduced_data):
    # Create an instance of the KMeansGaussianExperiment class
    kge = KMeansGaussianExperiment(reduced_data)

    tuner, study_silh = kge.tune_hyperparameter(choice, n_trials=25) # 25-30 minutes
    print(f"Best parameters: {study_silh.best_trial.params}")
    print(f"Best value: {study_silh.best_trial.value}")
    tuner.visualize_study(study_silh)

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
    y_prob = cexp.model.predict_proba(cexp.X_test)

    ceva = ClassificationEvaluator(cexp.model, cexp.y_test, y_pred)
    ceva.display_report()
    top_prod_proba_per_cat = ceva.get_top_produ_per_cat(df_cat_proba, cat_col, CATEGORIES, COUNT)
    best_cat_df = ceva.get_cat_ratios(df_cat_proba, cat_col, CATEGORIES)

    print()
    print(f'IDs of top {COUNT} products')
    for k, v in top_prod_proba_per_cat.items():
        print(f'in {k}')
        print(combined_df.iloc[v.index]['id'].to_list())

    top_produ_df = ceva.get_top_produ(df_pp, COUNT)
    print()
    print(f'Top {COUNT} products in all categories')
    print(combined_df.iloc[top_produ_df.index][['id', 'category', 'name']].join(top_produ_df.to_frame()))
    ceva.plot_confusion_matrix()
    ceva.visualize_best_category(best_cat_df)
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

