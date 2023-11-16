import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import json


DATASET_DIR = '../dataset/'
MODEL_DIR = '../models/'
SEED = 42

fn_dataset = 'dataset.csv'
fn_model_metrics = 'linear_model_metrics.json'
fn_model_pickle = 'reg_linear_model.pkl'
fn_lbgm_model_pickle = 'reg_lgbm_model.pkl'
fn_one_hot_encoding_pickle = 'one_hot_encoding.pkl'


def save_metrics (metrics, filename):
    
    with open(filename, "w") as f:
        json.dump(metrics, f)
    
def save_model (model, filename):
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    

def get_dataset(sample = None):

    data_set_path = DATASET_DIR + fn_dataset
    ds = pd.read_csv(data_set_path)

    if (sample != None):
        ds = ds.sample(sample)
    
    X = ds.drop('charges', axis=1)
    y = ds['charges']
    
    return X, y

def split_dataset(X, y):
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.1, random_state = SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = SEED)
    return  X_train, y_train, X_val, y_val, X_test, y_test

def encode_data(X, label_columns, model = None):
    
    #If the model is empty, I need to train it
    if (model == None):
        onehot_encoder = OneHotEncoder()
        onehot_encoder.fit(X[label_columns])
    #if not, I will just use it
    else:
        onehot_encoder = model
        
    categories = onehot_encoder.categories_
    nombres_columnas = [f'{cols}_{cat}' for cols, cats in zip(label_columns, categories) for cat in cats]

    # Creates a new DataFrame with the encodings
    X_encoded = pd.DataFrame(onehot_encoder.transform(X[label_columns]).toarray())
    X_encoded.columns = nombres_columnas

    #Add the new columns to the resul dataset
    X.reset_index(drop=True, inplace=True)
    X_encoded.reset_index(drop=True, inplace=True)
    X_combined = pd.concat([X, X_encoded], axis=1)
        
    #Removes the label columns (otherwise the regression wont work) 
    X_combined = X_combined.drop(label_columns, axis=1)

    return onehot_encoder, X_combined

def calculate_metrics(y_pred, y_test):
    
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"r2: {r2}")
    print(f"mae: {mae}")
    print(f"mse: {mse}")

    return r2, mse, mae

def encode_categorical_data(X_train, X_val, X_test):
    
    label_columns = ['sex', 'smoker', 'region']
    onehot_encoder, X_train = encode_data(X_train, label_columns)
    onehot_encoder, X_test = encode_data(X_test, label_columns, onehot_encoder)
    onehot_encoder, X_val = encode_data(X_val, label_columns, onehot_encoder)
    
    return onehot_encoder, X_train, X_val, X_test
    

def main():

    #Here execution starts 
    #Get the data
    X, y = get_dataset()

    #Split in train, valdation and test
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset (X, y)

    #Encodes the categorical data 
    onehot_encoder, X_train, X_val, X_test = encode_categorical_data(X_train, X_val, X_test)

    #Fits the model
    reg_linear_all = LinearRegression()
    reg_linear_all.fit(X_train, y_train)

    # calculates the predictions
    y_pred = reg_linear_all.predict(X_test)
    r2, mse, mae = calculate_metrics(y_pred, y_test)
    linear_model_metrics =  MODEL_DIR + fn_model_metrics

    save_metrics (dict({'model': 'linear_regression', 'r2': r2, 'mse': mse, 'mae': mae}),
                  linear_model_metrics)

    linear_model_path = MODEL_DIR + fn_model_pickle
    one_hot_encoding_path = MODEL_DIR + fn_one_hot_encoding_pickle

    # Saves the one_hot_encoder in a file
    save_model (onehot_encoder, one_hot_encoding_path)

    # Saves the linear model in a file
    save_model (reg_linear_all, linear_model_path)

    print(f'linear model saved in {linear_model_path}')
    print(f'one hot encoder model saved in {linear_model_path}')

if __name__ == "__main__":
  main()

