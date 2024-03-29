import joblib

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
        if cls._proccessor is None:
            proccessor_path = "pipline/preprocessor.pkl"
            cls._proccessor = joblib.load(proccessor_path)
        return cls._proccessor

    @classmethod
    def load_model(cls):
        # Load the model
        model_path = 'pipline/your_pipeline.pkl'
        pip = joblib.load(model_path)
        return pip


    
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
      
    
