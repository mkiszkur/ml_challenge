import numpy as np
import pandas as pd 
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
# evaluate logistic regression on the breast cancer dataset with an ordinal encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pickle
from training import *
from  training_lgbm import *

#DATASET_DIR = '../dataset/'
#MODEL_DIR = '../model/'


fn_lbgm_model_pickle = MODEL_DIR + 'reg_lgbm_model.pkl'
fn_one_hot_encoding_pickle = MODEL_DIR + 'one_hot_encoding.pkl'


X, y = get_dataset(10)

# Cargar el modelo desde el archivo pickle
with open(fn_lbgm_model_pickle, 'rb') as file:
    lgbm_loaded = pickle.load(file)

# Cargar el modelo desde el archivo pickle
with open(fn_one_hot_encoding_pickle, 'rb') as file:
    onehot_encoder =  pickle.load(file)
    

label_columns = ['sex', 'smoker', 'region']
onehot_encoder, X_encoded = encode_data(X, label_columns, onehot_encoder)


# Predicts on the sample
y_pred = lgbm_loaded.predict(X_encoded)

       
r2, mse, mae = calculate_metrics(y_pred, y)
save_metrics (dict({'model': 'lgbm_regression', 'r2': r2, 'mse': mse, 'mae': mae}),
                MODEL_DIR + fn_model_metrics)

