import pandas as pd
import numpy as np

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
# labeler = BinaryLabeler(df, 'value') # Default threshold is median
# print(labeler.get_threshold())
# df['label'] = labeler.get_labels()
# print(df)