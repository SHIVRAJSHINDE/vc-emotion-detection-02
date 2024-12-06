import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.main import utils

class FeatureEngineering:
    def __init__(self, params_path: str, train_data_path: str, test_data_path: str, output_path: str):
        """Initialize the FeatureEngineering class."""
        self.params_path = params_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.output_path = output_path

        self.logger = utils._setup_logger(self)
        self.params = utils.load_yaml_File(self,params_path)

        # Retrieve max_features parameter from the loaded YAML
        self.max_features = self.params['feature_engineering']['max_features']


    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            df.fillna('', inplace=True)  # Handle missing values
            self.logger.debug('Data loaded and NaNs filled from %s', file_path)
            return df
        except pd.errors.ParserError as e:
            self.logger.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error occurred while loading the data: %s', e)
            raise

    def apply_count_vectorizer(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
        """Apply CountVectorizer to the data."""
        try:
            vectorizer = CountVectorizer(max_features=self.max_features)

            X_train = train_data['content'].values
            y_train = train_data['sentiment'].values
            X_test = test_data['content'].values
            y_test = test_data['sentiment'].values

            X_train_bow = vectorizer.fit_transform(X_train)
            X_test_bow = vectorizer.transform(X_test)

            # Convert sparse matrices to DataFrames
            train_df = pd.DataFrame(X_train_bow.toarray())
            train_df['label'] = y_train

            test_df = pd.DataFrame(X_test_bow.toarray())
            test_df['label'] = y_test

            self.logger.debug('Bag of Words applied and data transformed')
            return train_df, test_df
        except Exception as e:
            self.logger.error('Error during Bag of Words transformation: %s', e)
            raise

    def save_data(self, df: pd.DataFrame, file_path: str) -> None:
        """Save the dataframe to a CSV file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            self.logger.debug('Data saved to %s', file_path)
        except Exception as e:
            self.logger.error('Unexpected error occurred while saving the data: %s', e)
            raise

    def run(self):
        """Execute the full feature engineering pipeline."""
        try:
            # Load training and test data
            train_data = self.load_data(self.train_data_path)
            test_data = self.load_data(self.test_data_path)

            # Apply CountVectorizer to the data
            train_df, test_df = self.apply_count_vectorizer(train_data, test_data)

            # Save processed data
            self.save_data(train_df, os.path.join(self.output_path, "train_tfidf.csv"))
            self.save_data(test_df, os.path.join(self.output_path, "test_tfidf.csv"))

        except Exception as e:
            self.logger.error('Failed to complete the feature engineering process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    # Initialize the FeatureEngineering class and run the pipeline
    feature_engineering = FeatureEngineering(
        params_path='params.yaml',
        train_data_path='./data/interim/train_processed.csv',
        test_data_path='./data/interim/test_processed.csv',
        output_path='./data/processed'
    )
    feature_engineering.run()
