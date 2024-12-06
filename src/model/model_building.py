import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from src.utils.main import utils


class ModelBuilder:


    def __init__(self, params_path: str, data_path: str, model_save_path: str, log_file: str):
        self.params_path = params_path
        self.data_path = data_path
        self.model_save_path = model_save_path

        self.logger = utils._setup_logger(self)
        self.params = utils.load_yaml_File(self,params_path)

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

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
        """Train the Gradient Boosting model."""
        try:
            clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
            clf.fit(X_train, y_train)
            self.logger.debug('Model training completed')
            return clf
        except Exception as e:
            self.logger.error('Error during model training: %s', e)
            raise

    def save_model(self, model) -> None:
        """Save the trained model to a file."""
        try:
            with open(self.model_save_path, 'wb') as file:
                pickle.dump(model, file)
            self.logger.debug('Model saved to %s', self.model_save_path)
        except Exception as e:
            self.logger.error('Error occurred while saving the model: %s', e)
            raise

    def build_and_save_model(self):
        """The full process of loading data, training, and saving the model."""
        try:
            params = self.params['model_building']

            train_data = self.load_data()
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values

            clf = self.train_model(X_train, y_train, params)
            self.save_model(clf)
        except Exception as e:
            self.logger.error('Failed to complete the model building process: %s', e)
            print(f"Error: {e}")

# Entry point
if __name__ == '__main__':
    model_builder = ModelBuilder(
        params_path='params.yaml',
        data_path='./data/processed/train_tfidf.csv',
        model_save_path='models/model.pkl',
        log_file='model_building_errors.log'
    )
    model_builder.build_and_save_model()
