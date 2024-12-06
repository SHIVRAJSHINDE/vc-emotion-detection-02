import numpy as np
import pandas as pd
import os
import yaml
import logging

class utils:

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
    
    def load_yaml_File(self,file_Path) -> dict:

        self.logger = utils._setup_logger(self)

        """Load parameters from a YAML file."""
        try:

            with open(file_Path, 'r') as file:
                params = yaml.safe_load(file)
            self.logger.debug('Parameters retrieved from %s', file_Path)
            return params

        except FileNotFoundError:
            self.logger.error('File not found: %s', file_Path)
            raise
        except yaml.YAMLError as e:
            self.logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise

    