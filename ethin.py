import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from sklearn.pipeline import make_pipeline
from tqdm import tqdm

class BinaryLabeler:
    """
    A class used to label records in a DataFrame as binary classes (0 and 1)
    based on a specified metric and threshold.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data to be labeled.
    metric : str
        The column name of the metric in the DataFrame used for labeling.
    threshold : callable or float
        A function or a float value used as the threshold to split the classes.
        Default is np.median.

    Methods
    -------
    get_labels():
        Returns a pandas Series of binary labels (0 and 1) based on the specified metric and threshold.
    get_threshold():
        Returns the computed threshold value used for labeling.
    """

    def __init__(self, dataframe: pd.DataFrame, metric: str, threshold=np.median):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            The input DataFrame containing the data to be labeled.
        metric : str
            The column name of the metric in the DataFrame used for labeling.
        threshold : callable or float, optional
            A function or a float value used as the threshold to split the classes.
            Default is np.median.
        """
        self.dataframe = dataframe
        self.metric = metric
        self.threshold = threshold

    def get_threshold(self) -> float:
      """
      Returns the computed threshold value used for labeling.

      Returns
      -------
      float
          The threshold value.
      """
      if callable(self.threshold):
          return self.threshold(self.dataframe[self.metric])
      else:
          return self.threshold

    def get_labels(self) -> pd.Series:
        """
        Returns a pandas Series of binary labels (0 and 1) based on the specified metric and threshold.

        Returns
        -------
        pd.Series
            A Series of binary labels (0 and 1).
        """
        threshold_value = self.get_threshold()
        labels = (self.dataframe[self.metric] >= threshold_value).astype(int)
        return labels


# Example usage:
# df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
# labeler = BinaryLabeler(df, 'value')
# threshold_value = labeler.get_threshold()
# labels = labeler.get_label()
# print("Threshold:", threshold_value)
# print("Labels:", labels)

class LogRegExperiment:
    """
    A class to perform logistic regression with elastic net regularization
    and cross-validation.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing features and target.
    feature_columns : list
        List of column names to be used as features.
    target_column : str
        The column name of the target variable.

    Methods
    -------
    run_experiment():
        Performs logistic regression with elastic net regularization and
        returns probability scores for classification results.
    """

    def __init__(self, dataframe: pd.DataFrame, feature_columns: list, target_column: str):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            The input DataFrame containing features and target.
        feature_columns : list
            List of column names to be used as features.
        target_column : str
            The column name of the target variable.
        """
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column

    def run_experiment(self):
        """
        Performs logistic regression with elastic net regularization and
        returns probability scores for classification results.

        Returns
        -------
        y_prob : np.ndarray
            The probability scores for the positive class.
        """
        # Extract features and target
        X = self.dataframe[self.feature_columns]
        y = self.dataframe[self.target_column]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Perform logistic regression with elastic net regularization
        model = LogisticRegressionCV(
            cv=5,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.1, 0.5, 0.9],
            max_iter=10000
        )
        model.fit(X_train, y_train)

        # Predict probability scores
        y_prob = model.predict_proba(X_test)[:, 1]

        # Print classification report
        y_pred = model.predict(X_test)
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

        return y_prob

# Example usage:
# df = pd.read_csv('your_data.csv')  # Load your data
# experiment = LogisticRegressionExperiment(df, ['feature1', 'feature2'], 'target')
# probabilities = experiment.run_experiment()
# print(probabilities)import pandas as pd

class LogRegExperiment:
    """
    A class to perform logistic regression with elastic net regularization
    and cross-validation.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing features and target.
    feature_columns : list
        List of column names to be used as features.
    target_column : str
        The column name of the target variable.
    best_model : sklearn estimator
        The best logistic regression model found by GridSearchCV.
    y_prob : np.ndarray
        The probability scores for the positive class.

    Methods
    -------
    run_experiment():
        Performs logistic regression with elastic net regularization and
        returns probability scores for classification results.
    """

    def __init__(self, dataframe: pd.DataFrame, feature_columns: list, target_column: str):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            The input DataFrame containing features and target.
        feature_columns : list
            List of column names to be used as features.
        target_column : str
            The column name of the target variable.
        """
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.best_model = None
        self.y_prob = None

    def run_experiment(self):
        """
        Performs logistic regression with elastic net regularization and
        returns probability scores for classification results.

        Returns
        -------
        y_prob : np.ndarray
            The probability scores for the positive class.
        """
        # Extract features and target
        X = self.dataframe[self.feature_columns]
        y = self.dataframe[self.target_column]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Define the logistic regression model with elastic net regularization
        model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)

        # Define the parameter grid for GridSearchCV
        param_grid = {'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

        # Initialize GridSearchCV with tqdm for progress tracking
        grid_search = GridSearchCV(model, param_grid, cv=5, verbose=0, n_jobs=-1)

        # Wrap the fit method with tqdm to show progress
        # with tqdm(total=len(param_grid['l1_ratio']) * 5, desc="Grid Search Progress", unit="iteration") as pbar:
        #     def update_bar(*args, **kwargs):
        #         pbar.update(1)

        # grid_search.fit(X_train, y_train, callback=update_bar)
        grid_search.fit(X_train, y_train)

        # Save the best model
        self.best_model = grid_search.best_estimator_

        # Predict probability scores FOR WHOLE DATAFRAME using the best model
        self.y_prob = self.best_model.predict_proba(X_scaled)[:, 1]
        print(X_scaled.shape)

        # Print classification report
        y_pred = self.best_model.predict(X_test)
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

# Example usage:
# df = pd.read_csv('your_data.csv')  # Load your data
# experiment = LogRegExperiment(df, ['feature1', 'feature2'], 'target')
# probabilities = experiment.run_experiment()
# print(probabilities)

class TopEvaluator:
    """
    A class to evaluate and compare the top n values from two columns of a pandas DataFrame.

    Attributes:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column1 : str
        The name of the first column to evaluate.
    column2 : str
        The name of the second column to evaluate.
    n : int
        The number of top values to select from each column.
    compare_table : pd.DataFrame
        A DataFrame comparing the top n values from both columns.
    match_percentage : float
        The percentage of top n values that match between the two columns.

    Methods:
    --------
    evaluate():
        Selects the top n values from the specified columns and calculates the match percentage.
    """

    def __init__(self, df, column1, column2, n):
        """
        Initializes the TopEvaluator with the DataFrame, column names, and number of top values.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data.
        column1 : str
            The name of the first column to evaluate.
        column2 : str
            The name of the second column to evaluate.
        n : int
            The number of top values to select from each column.
        """
        self.df = df
        self.column1 = column1
        self.column2 = column2
        self.n = n
        self.compare_table = pd.DataFrame()
        self.match_percentage = 0.0

    def evaluate(self):
        """
        Selects the top n values from the specified columns and calculates the match percentage.
        """
        # Select top n values from each column
        top_n_col1 = self.df.nlargest(self.n, self.column1)
        top_n_col2 = self.df.nlargest(self.n, self.column2)

        # Combine the indices
        combined_indices = top_n_col1.index.union(top_n_col2.index)

        # Create a comparison table
        self.compare_table = self.df.loc[combined_indices, [self.column1, self.column2]]

        # Calculate match percentage
        matches = top_n_col1.index.intersection(top_n_col2.index)
        self.match_percentage = (len(matches) / self.n) * 100

# Example usage:
# data = {
#     'Column1': [10, 20, 15, 30, 25],
#     'Column2': [100, 200, 150, 300, 250]
# }
# df = pd.DataFrame(data)

# evaluator = TopEvaluator(df, 'Column1', 'Column2', 3)
# evaluator.evaluate()

# print("Comparison Table:")
# print(evaluator.compare_table)
# print(f"Match Percentage: {evaluator.match_percentage}%")