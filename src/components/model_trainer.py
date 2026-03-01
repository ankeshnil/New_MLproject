# here er train different model 

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from src.utils import evaluate_model
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTraningConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTraningConfig()
    def initiate_model_traning(self,train_array, test_array):
        logging.info("Initiate_model_traing_function")
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            models={
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }
            params = {

    "Decision Tree": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10]
    },

    "K-Neighbors Regressor": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },

    "Random Forest Regressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    },

    "Linear Regression": {},

    "XGBRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4]
    },

    "AdaBoost Regressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1]
    },

    "Ridge": {
        "alpha": [0.01, 0.1, 1, 10]
    },

    "Lasso": {
        "alpha": [0.01, 0.1, 1]
    }
}
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test, y_test= y_test,
                                             models= models, param = params) # this fun. create in utils.py
            
            logging.info("Model evaluation report store")
            
            # to get the best model score            
            best_model_score = max(sorted(model_report.values())) 
            #to get the best model name 
            best_model_name =  list(model_report.keys())[
                            list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]  # it return RandomForestRegressor() like this
             
            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info("Best Model Found")
            
            save_object(   # we write the function in utils.py
               file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model  # → what to save
            ) 
            
            predicted = best_model.predict(X_test)
            model_score = r2_score(y_test, predicted)
            
            logging.info("R2_scored returend")
            
            return model_score
                      
        except Exception as e:
            print("Original Error:", e)
            raise CustomException(e,sys)



