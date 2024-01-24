from mufpo.etl import Pipe
from .model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import os

TARGET_COL = 'nav/unit'
MODEL_PATH = "/Users/vitvaraaravithya/Documents/final_project/ml/model/test"
PERIOD_PREDICT = 30
SEASONAL = True
M = 12

def fit_arima_model(time_series, seasonal=True, m=12):
    arima_model = ARIMA(m, seasonal)
    arima_model.fit(time_series)
    evaluation_results = arima_model.eval()

    return arima_model, evaluation_results

def train(dataset, seasonal, m, target_col, model_path):
    _, train, _ = dataset
    model, eval = fit_arima_model(train[target_col], seasonal=seasonal, m=m)
    return save_model(eval, model, dataset, model_path)

def save_model__inner(eval, model, dataset, save_path):
    fund_name, _, _ = dataset
    
    joblib.dump(model, os.path.join(save_path, f'{fund_name}.joblib'))
    return eval, dataset

def save_model(eval, model, dataset, model_path):
    return save_model__inner(eval, model, dataset, model_path)



class Trainner:
    def __init__(self, datasets, model_path, seasonal, m, target_col):
        self.datasets = datasets
        self.model_path = model_path
        self.seasonal = seasonal
        self.m = m
        self.target_col = target_col

    def map_train(self, x):
        return list(
            map(
                lambda x: train(x, self.seasonal, self.m, self.target_col, self.model_path), x
            )
        )
    
    def map_save_model(self, x):
        return list(map(save_model, *x))

    def __call__(self):
        return (
            Pipe(self.datasets)
            | self.map_train
            | self.map_save_model
        )