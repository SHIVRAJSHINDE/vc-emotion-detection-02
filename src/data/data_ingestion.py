import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

class DataIngestion:
    def __init__(self, params_path: str, data_url: str, data_path: str):
        self.params_path = params_path
        self.data_url = data_url
        self.data_path = data_path
        self.logger = self._setup_logger()
        self.params = self.load_params()
        self.test_size = self.params['data_ingestion']['test_size']

    def _setup_logger(self):
        """Setup logger configuration"""
        logger = logging.getLogger('data_ingestion')
        logger.setLevel('DEBUG')

        console_handler = logging.StreamHandler()
        console_handler.setLevel('DEBUG')

        file_handler = logging.FileHandler('errors.log')
        file_handler.setLevel('ERROR')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

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

            df = pd.read_csv(self.data_url)
            self.logger.debug('Data loaded from %s', self.data_url)
            return df

        except pd.errors.ParserError as e:
            self.logger.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error occurred while loading the data: %s', e)
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            df.drop(columns=['tweet_id'], inplace=True)
            final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
            final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
            self.logger.debug('Data preprocessing completed')
            return final_df

        except KeyError as e:

            self.logger.error('Missing column in the dataframe: %s', e)
            raise

        except Exception as e:

            self.logger.error('Unexpected error during preprocessing: %s', e)
            raise

    def save_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Save the train and test datasets."""
        try:

            raw_data_path = os.path.join(self.data_path, 'raw')
            os.makedirs(raw_data_path, exist_ok=True)
            train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
            test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
            self.logger.debug('Train and test data saved to %s', raw_data_path)

        except Exception as e:
            self.logger.error('Unexpected error occurred while saving the data: %s', e)
            raise

    def run(self):
        """Main method to execute data ingestion pipeline."""
        try:

            df = self.load_data()
            final_df = self.preprocess_data(df)
            train_data, test_data = train_test_split(final_df, test_size=self.test_size, random_state=42)
            self.save_data(train_data, test_data)

        except Exception as e:
            self.logger.error('Failed to complete the data ingestion process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    # Initialize and run the DataIngestion pipeline
    params_path='params.yaml', 
    data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv', 
    data_path='./data'
    
    data_ingestion = DataIngestion(params_path=params_path, data_url=data_url, data_path=data_path)

    data_ingestion.run()
