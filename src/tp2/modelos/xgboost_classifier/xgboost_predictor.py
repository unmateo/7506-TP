#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
from datos import FEATURES_DISPONIBLES
from modelo import Modelo

pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[2]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[58]:


class XGBoost(Modelo):
    """
        Este modelo lo vamos a usar para predecir valores
        faltantes en los sets de datos.
    """
    
    excluir = {
        "idzona",
        "ciudad",
        "provincia",
        "precio",
        "precio_metro_total",
        "precio_metro_cubierto",
        "gps", "lat", "lng"
    }
    features = FEATURES_DISPONIBLES - excluir
        
    def __init__(self, feature):
        """
        """
        assert feature in self.features
        super().__init__(feature)


    def cargar_datos(self, seed=42):
        """
            Junto los 3 dataframes y armo nuevos sets:
            train y test con las publicaciones que tengan
            el dato a predecir, submit con las que no lo tengan.
        """
        super().cargar_datos(self.features)
        todos = self.test_data            .append(self.train_data, sort=False)            .append(self.submit_data, sort=False)            .drop(columns=[
                'fecha',
                'titulo',
                'descripcion']
            )
        todos = self.one_hot_encode(todos, ['tipodepropiedad'])
        self.clases_feature = len(todos[self.feature].unique())
        self.submit_data = todos.loc[todos[self.feature].isna()]
        con_feature = todos.loc[todos[self.feature].notna()]
        train, test = train_test_split(con_feature, random_state=seed)
        self.train_data = train
        self.test_data = test
        return True
    

    def _split_data_label(self, df, label):
        data = df.loc[:, df.columns != label]
        label = df[label].values
        return data, label
    
    @Modelo.cronometrar()
    def entrenar(self, params=None):
        """
        """
        if not params:
            params = {
                'max_depth': 2,
                'eta': 1,
                'objective': 'multi:softmax',
                'num_class': self.clases_feature,
                'nthread': 4,
                'eval_metric': 'merror'
            }
        train_data, train_label = self._split_data_label(self.train_data, self.feature)
        dtrain = xgb.DMatrix(train_data, label=train_label)
        self.predictor = xgb.train(params, dtrain)
        super().entrenar()
        return True
    
    @Modelo.cronometrar()
    def predecir(self, df):
        """
            to_bool: bool
                Transformar la columna target (0,1] a bool.
                Lo hace poniendo un límite en 0.5
        """
        data = df.copy()
        predict_data, predict_label = self._split_data_label(data, self.feature)
        dpredict = xgb.DMatrix(predict_data)
        data["target"] = self.predictor.predict(dpredict)
        return data

    
    def puntuar(self, real, prediccion):
        """
            
        """
        return (real == prediccion).value_counts(normalize=True)[True]
        


# In[63]:


def test(feature):
    modelo = XGBoost(feature=feature)
    modelo.cargar_datos()
    modelo.entrenar()
    print(modelo.validar())
    return modelo


# In[ ]:


def tests():
    modelo_garage = test('garages')
    modelo_banos= test('banos')
    modelo_habitaciones = test('habitaciones')
    return modelo_garage, modelo_banos, modelo_habitaciones

