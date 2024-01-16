import joblib
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
class Pipeline:
    _pipeline = None  # Class property to store the pipeline
    _proccessor=None
    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            cls._pipeline= cls.load_model()
        return cls._pipeline
    @classmethod
    def get_proccessor(cls):
        try:
            if cls._proccessor is None:
                current_folder = os.path.dirname(os.path.abspath(__file__))
                proccessor_path = os.path.join(current_folder, 'preprocessor.pkl')
                cls._proccessor = joblib.load(proccessor_path)
            return cls._proccessor
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            print(f"Current files in the directory: {os.listdir(current_folder)}")
            raise

    @classmethod
    def load_model(cls):
        try:
            current_folder = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_folder, 'pipeline.pkl')
            print(model_path)
            pip = joblib.load(model_path)
            return pip
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            print(f"Current files in the directory: {os.listdir(current_folder)}")
            raise

    
    @classmethod
    def predict(cls, input_data):
        model = cls.get_pipeline()
        return model.predict(input_data)[0]

    @classmethod
    def visualize_tree(cls):
        classifier = cls.get_pipeline().named_steps['classifier']
        fig, ax = plt.subplots(figsize=(23, 19))
        plot_tree(classifier,
                  feature_names=cls.get_proccessor().get_feature_names_out(['outlook', 'temp', 'humidity', 'windy']),
                  class_names=['No', 'Yes'],
                  filled=True,
                  rounded=True,
                  impurity=False,
                  proportion=True,
                  precision=2,
                  ax=ax,
                  fontsize=16)
        ax.set_title("Decision Tree for Tennis Decision Making")
        return fig
      
    
