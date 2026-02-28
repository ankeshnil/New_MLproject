# it contain all the code that help to read the data . Here we read the data from different data Scourse
# And here aslo we do train test split

import os  # it can do Create folders, Join file paths properly, check if file exist 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # It automatically creates: __init__, __repr__, __eq__
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTraningConfig

@dataclass
class DataIngestionConfig: # here we create the file path
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # when we call Dataingestion ,ingestion_config store all the path
        
    def initiate_data_ingestion(self):
        logging.info("enterd the data ingestion methodd or components")
        try:
            df = pd.read_csv('Notebook/Data/StudentsPerformance.csv')
            logging.info("Read the dataset from data frame")
            
            # It only creates the folder where train.csv will be saved. not create the train.csv file
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # to save the raw data inside the file that we created avobe    df.to_csv(path-> where to save, index=False, header=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train_test_spli initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # to save the raw data inside the file that we created avobe 
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                # we return it beacuse at the next setp in data transformation we just get thos train and test data
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=="__main__":
    obj= DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_train = ModelTrainer()
    print(model_train.initiate_model_traning(train_arr, test_arr))