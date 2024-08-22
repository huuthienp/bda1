import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class RandomForestExperiment:
    def __init__(self):
        self.X_test = None
        self.y_test = None

    #TODO: Just simple preprocess, remove when common preprocessing decided
    def split_and_fit(self, X, y) -> RandomForestClassifier:
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        params_RF = {'max_depth': 6, 'criterion': 'log_loss', 'min_samples_split': 2, 'max_features': 'log2', 'max_samples': 0.7264084469432243}
        RF_model_tuned = RandomForestClassifier(random_state=42, **params_RF, n_estimators = 500)
        return RF_model_tuned.fit(X_train, y_train)

    def plot_confusion_matrix(self, y_test: np.array, RF_predictions: np.array, save=False) -> None:
        # Calculate confusion matrix
        confusion_matrix_df = pd.DataFrame(data=confusion_matrix(y_test, RF_predictions), 
                                       index=["True Good", "True Bad"], 
                                       columns=["Predicted Good", "Predicted Bad"])
    
        # Plot confusion matrix
        fig = plt.figure()
        sns.heatmap(confusion_matrix_df, annot=True)
        plt.show()
        if save:
            plt.savefig('ConfusionMatrix_RandomForest.png')
 
    def display_accuracy_on_testset(self, y_test: np.array, RF_predictions: np.array):
        accuracy = accuracy_score(y_test, RF_predictions)
        print(f"The accuracy of hyperparamater tuned is: {accuracy}")
    
    def get_top_products_per_cat(self, RF_model: RandomForestClassifier, X, category_column, categories, how_many_at_top) -> dict:
        top_products = {}
        for cat in categories:
            # Filter data for the current category
            filtered_data = X[category_column == cat]
            # Predict probabilities for the current category
            proba_data = RF_model.predict_proba(filtered_data)
            # Convert probabilities to DataFrame
            proba_df = pd.DataFrame(proba_data, columns=['Class_0', 'Class_1'], index=filtered_data.index)
            top_idx = proba_df['Class_1'].nlargest(how_many_at_top).index
            top_products[cat] = proba_df.loc[top_idx]
        return top_products    

    def display_category_ratios(self, RF_model: RandomForestClassifier, X, category_column, categories):
        predictions = {}
    
        for cat in categories:
            # Filter data for the current category and make predictions
            filtered_data = X[category_column == cat]
            predictions[cat] = RF_model.predict(filtered_data)
    
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

# if __name__ == '__main__':
#     """
#     EXAMPLE USAGE FOR MAIN FILE
#     """
#     preprocessed_data = add_category_information("preprocessed_data.csv", "data.csv")
#     X_train, X_test, y_train, y_test = split_train_test(preprocessed_data)
#     RF_model = fit_to_train_set(X_train, y_train)
#     RF_predictions = RF_model.predict(X_test)
#     plot_confusion_matrix(y_test, RF_predictions)
#     display_accuracy_on_testset(y_test, RF_predictions)

#     #Below is code to define top 10 products and categories
#     df = display_category_ratios(preprocessed_data, RF_model)


