import joblib
import os

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self, model_path, fund_name):
        return joblib.load(os.path.join(model_path, f'{fund_name}.joblib'))