from .object_model.data_loader import DataLoader
from .object_model.preprocessor import Preprocessor

class TrainAndValidate:
    def __init__(self, daily_nav, target_col, seasonal, m, model_save_path):
        self.target_col = target_col
        self.seasonal = seasonal
        self.m = m
        self.model_save_path = model_save_path
        self.daily_nav = daily_nav

    def run(self):
        data_loader = DataLoader(self.daily_nav)
        preprocessor = Preprocessor(data_loader)
        
