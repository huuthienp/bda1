import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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
# print(probabilities)