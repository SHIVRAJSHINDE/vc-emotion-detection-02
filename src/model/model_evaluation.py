import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
from src.utils.main import utils


class ModelEvaluator:
    def __init__(self, model_path: str, data_path: str, metrics_save_path: str, log_file: str):
        self.model_path = model_path
        self.data_path = data_path
        self.metrics_save_path = metrics_save_path
        self.logger = utils._setup_logger(self)

    def load_model(self):
        """Load the trained model from a file."""
        try:
            with open(self.model_path, 'rb') as file:
                model = pickle.load(file)
            self.logger.debug('Model loaded from %s', self.model_path)
            return model
        except FileNotFoundError:
            self.logger.error('File not found: %s', self.model_path)
            raise
        except Exception as e:
            self.logger.error('Unexpected error occurred while loading the model: %s', e)
            raise

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            self.logger.debug('Data loaded from %s', self.data_path)
            return df
        except pd.errors.ParserError as e:
            self.logger.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error occurred while loading the data: %s', e)
            raise

    def evaluate_model(self, clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and return the evaluation metrics."""
        try:
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)

            metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            self.logger.debug('Model evaluation metrics calculated')
            return metrics_dict
        except Exception as e:
            self.logger.error('Error during model evaluation: %s', e)
            raise

    def save_metrics(self, metrics: dict) -> None:
        """Save the evaluation metrics to a JSON file."""
        try:
            with open(self.metrics_save_path, 'w') as file:
                json.dump(metrics, file, indent=4)
            self.logger.debug('Metrics saved to %s', self.metrics_save_path)
        except Exception as e:
            self.logger.error('Error occurred while saving the metrics: %s', e)
            raise

    def evaluate_and_save_metrics(self):
        """The full process of loading the model, evaluating it, and saving metrics."""
        try:
            clf = self.load_model()
            test_data = self.load_data()

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = self.evaluate_model(clf, X_test, y_test)
            self.save_metrics(metrics)
        except Exception as e:
            self.logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

# Entry point
if __name__ == '__main__':
    model_evaluator = ModelEvaluator(
        model_path='./models/model.pkl',
        data_path='./data/processed/test_tfidf.csv',
        metrics_save_path='reports/metrics.json',
        log_file='model_evaluation_errors.log'
    )
    model_evaluator.evaluate_and_save_metrics()
