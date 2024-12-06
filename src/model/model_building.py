import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

class ModelBuilder:
    def __init__(self, params_path: str, data_path: str, model_save_path: str, log_file: str):
        self.params_path = params_path
        self.data_path = data_path
        self.model_save_path = model_save_path
        
        # Initialize logger
        self.logger = logging.getLogger('model_building')
        self.logger.setLevel('DEBUG')

        console_handler = logging.StreamHandler()
        console_handler.setLevel('DEBUG')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel('ERROR')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def load_params(self) -> dict:
        """Load parameters from a YAML file."""
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            self.logger.debug('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            self.logger.error('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            self.logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
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
            params = self.load_params()['model_building']

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
