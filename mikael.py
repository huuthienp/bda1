import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
        print(f"The recall of {self.model.__class__.__name__} is: {recall:.2f}")    
        print()

    def get_top_produ_per_cat(self, df_cat, category_column, categories, count) -> dict:
        top_produ = {}
        for cat in categories:
            # Filter data for the current category
            filtered_data = df_cat[category_column == cat]
            # Predict probabilities for the current category
            proba_data = self.model.predict_proba(filtered_data.drop(columns='category'))
            # Convert probabilities to DataFrame
            proba_df = pd.DataFrame(proba_data, columns=['Class_0', 'Class_1'], index=filtered_data.index)
            top_idx = proba_df['Class_1'].nlargest(count).index
            top_produ[cat] = proba_df.loc[top_idx]
        return top_produ    

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

if __name__ == '__main__':
    CATEGORIES = ['shoes', 'women', 'house', 'men', 'accessories', 'bags', 'jewelry', 'kids', 'beauty']
    COUNT = 10
    from robbie import DataReader, DataPreProcesser
    # define path variables
    zip_file_dir = "A1_2024_Released.zip"
    csv_dir = "A1_2024_Unzip"

    # obtain the data
    data_reader = DataReader(zip_file_dir, csv_dir)
    combined_df = data_reader.combined_df
    dpp = DataPreProcesser(combined_df)
    df_pp = dpp.df

    df_cat = add_category_information(df_pp, combined_df)
    cat_col = df_cat['category']

    params_rf = {'max_depth': 6, 
                 'criterion': 'log_loss', 
                 'min_samples_split': 2, 
                 'max_features': 'log2', 
                 'max_samples': 0.7264084469432243,
                 'n_estimators': 500}

    rfce = ClassificationExperiment(1, params_rf)
    rfce.split_and_fit(df_pp)
    print(rfce.model)
    rf_y_pred = rfce.model.predict(rfce.X_test)

    rf_eval = ClassificationEvaluator(rfce.model, rfce.y_test, rf_y_pred)
    rf_eval.display_report()

    df_cat_proba = df_cat.drop(columns='target')
    rf_top_prod_proba_per_cat = rf_eval.get_top_produ_per_cat(df_cat_proba, cat_col, CATEGORIES, COUNT)
    rf_best_cat_df = rf_eval.get_cat_ratios(df_cat_proba, cat_col, CATEGORIES)

    rf_eval.plot_confusion_matrix()