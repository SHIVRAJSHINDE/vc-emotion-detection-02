import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

from src.utils.main import utils


class DataIngestion:
    def __init__(self, params_path: str, data_url: str, data_path: str):

        self.logger = utils._setup_logger(self)
        self.params = utils.load_yaml_File(self,params_path)

        self.params_path = params_path
        self.data_url = data_url
        self.data_path = data_path
       
        self.test_size = self.params['data_ingestion']['test_size']



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
    data_ingestion = DataIngestion(

        params_path='params.yaml',
        data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv',
        data_path='./data'


    )

    data_ingestion.run()



