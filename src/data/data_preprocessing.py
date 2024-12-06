import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils.main import utils


class TextNormalizer:
    def __init__(self, data_path: str, raw_data_path: str, processed_data_path: str):
        # Initializing paths and logger
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.logger = utils._setup_logger(self)
        

        
        # Downloading necessary NLTK data
        nltk.download('wordnet')
        nltk.download('stopwords')


    def lemmatization(self, text):
        """Lemmatize the text."""
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)

    def remove_stop_words(self, text):
        """Remove stop words from the text."""
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]
        return " ".join(text)

    def removing_numbers(self, text):
        """Remove numbers from the text."""
        text = ''.join([char for char in text if not char.isdigit()])
        return text

    def lower_case(self, text):
        """Convert text to lower case."""
        text = text.split()
        text = [word.lower() for word in text]
        return " ".join(text)

    def removing_punctuations(self, text):
        """Remove punctuations from the text."""
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        return text

    def removing_urls(self, text):
        """Remove URLs from the text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_small_sentences(self, df):
        """Remove sentences with less than 3 words."""
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan

    def normalize_text(self, df):
        """Normalize the text data."""
        try:
            df['content'] = df['content'].apply(self.lower_case)
            self.logger.debug('Converted to lower case')
            df['content'] = df['content'].apply(self.remove_stop_words)
            self.logger.debug('Stop words removed')
            df['content'] = df['content'].apply(self.removing_numbers)
            self.logger.debug('Numbers removed')
            df['content'] = df['content'].apply(self.removing_punctuations)
            self.logger.debug('Punctuations removed')
            df['content'] = df['content'].apply(self.removing_urls)
            self.logger.debug('URLs removed')
            df['content'] = df['content'].apply(self.lemmatization)
            self.logger.debug('Lemmatization performed')
            self.logger.debug('Text normalization completed')
            return df
        except Exception as e:
            self.logger.error('Error during text normalization: %s', e)
            raise

    def process_and_save_data(self):
        """Main method to process and save the data."""
        try:
            # Load the data
            train_data = pd.read_csv(os.path.join(self.raw_data_path, 'train.csv'))
            test_data = pd.read_csv(os.path.join(self.raw_data_path, 'test.csv'))
            self.logger.debug('Data loaded successfully')

            # Normalize the text data
            train_processed_data = self.normalize_text(train_data)
            test_processed_data = self.normalize_text(test_data)

            # Create processed data directory
            os.makedirs(self.processed_data_path, exist_ok=True)

            # Save processed data
            train_processed_data.to_csv(os.path.join(self.processed_data_path, "train_processed.csv"), index=False)
            test_processed_data.to_csv(os.path.join(self.processed_data_path, "test_processed.csv"), index=False)

            self.logger.debug(f'Processed data saved to {self.processed_data_path}')
        except Exception as e:
            self.logger.error('Failed to complete the data transformation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    # Set paths
    data_path = './data'
    raw_data_path = os.path.join(data_path, 'raw')
    processed_data_path = os.path.join(data_path, 'interim')

    # Initialize the TextNormalizer class
    text_normalizer = TextNormalizer(data_path=data_path, raw_data_path=raw_data_path, processed_data_path=processed_data_path)

    # Process and save the data
    text_normalizer.process_and_save_data()
