import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # We create this to save the preprocessor object (like StandardScaler, OneHotEncoder, ColumnTransformer) so that we can reuse it later during prediction.
    preprocessor_ob_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):  # we cretae this function to create our pkl file of data transformation 
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                        "gender",
                        "race/ethnicity",
                        "parental level of education",
                        "lunch",
                        "test preparation course",
                        ]
            
            # crating a pipeline for numarical feature to do imputation and standerdScaller
            num_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            # for categorical feature
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),   # optional 
                ]
            )
            logging.info("Categorical and Numerical encoding and scalinf is completed")
            
            # apply to dataset
            preprocessor = ColumnTransformer(
            transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ]
)
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
# ----------------------------------
# Initiate Data Transformation
# ----------------------------------
    
    # Now we do the data transfroamtion
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            # here our target columns is math score
        
            logging.info("Read train and Test Data Succesfully")
        
            preprocession_obj = self.get_data_transformer_obj()
        
            input_feature_train_df = train_data.drop(columns=["math score"], axis=1)  
            target_feature_train_df = train_data["math score"]
        
            input_feature_test_df = test_data.drop(columns=["math score"], axis=1)  
            target_feature_test_df = test_data["math score"]  
                
            input_feature_train_arr = preprocession_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocession_obj.transform(input_feature_test_df)
        
            logging.info("fit transfrom done on X_train and y_train")
        
        
            train_arr = np.c_[     # np.c  is a NumPy function used to concatenate arrays column-wise.
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[    
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
        
            # this function saves the pickle file. inside the folder path you defined in preprocessor_ob_file_path
            save_object(   # we write the function in utils.py
               file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocession_obj
            )
            
            logging.info("Save preprocessing object")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
    