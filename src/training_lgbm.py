from training import *
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from hyperopt import fmin, hp, tpe



fn_lbgm_model_pickle = 'reg_lgbm_model.pkl'
fn_model_metrics = 'lgbm_model_metrics.json'

def train_baseline_model(X_train, y_train, X_test, y_test):
        # Crear el modelo de regresión LightGBM
    model = lgb.LGBMRegressor()

    # Ajustar el modelo a los datos de entrenamiento
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de testing
    y_pred = model.predict(X_test)
   
    r2, mse, mae = calculate_metrics(y_pred, y_test)

    print("R2:", r2) 
    print("MAE:", mae)
    print("MSE:", mse)

    return model, r2, mae, mse

 
    
def main():
    
    def objective(params):

        print (params)
        # Configurar modelo con hiperparámetros candidatos 
        model = lgb.LGBMRegressor(**params)

        # Entrenar modelo
        model.fit(X_train, y_train)

        # Hacer predicciones en el conjunto de validación
        y_pred = model.predict(X_val)

        # Calcular RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        # Devolver el RMSE (minimizar)
        return rmse

    #Here execution starts 
    #Get the data
    X, y = get_dataset()

    #Split in train, valdation and test
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset (X,y )


    #Encodes the categorical data 
    onehot_encoder, X_train, X_val, X_test = encode_categorical_data(X_train, X_val, X_test)

    baseline_model, r2, mae, mse =  train_baseline_model(X_train, y_train, X_test, y_test)
   

    space = {
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        'num_leaves': hp.randint('num_leaves', 10, 50),
        'max_depth': hp.randint('max_depth', 5, 30),
        'min_child_samples': hp.randint('min_child_samples', 10, 50),
    }


    print("### started hyperparameters started ###\n\n")


    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, verbose=False)

    print("\n\n\n### hyperparameters search finished ###")
    print("... found the following hyperparameters ...\n\n")

    print(best)


    model = lgb.LGBMRegressor()
    model.learning_rate = best['learning_rate']
    model.max_depth = best['max_depth']
    model.min_child_samples = best['min_child_samples']
    model.num_leaves = best['num_leaves']


    # Ajustar el modelo a los datos de entrenamiento
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de testing
    y_pred = model.predict(X_test)
    print("... calculating performance metrics ...\n\n")

    r2_2, mse_2, mae_2 = calculate_metrics(y_pred, y_test)

    # Saves the linear model in a file
    print("... model and one hot encoder saved ...\n")
    model_path = MODEL_DIR + fn_lbgm_model_pickle
    save_model(model, model_path)
    print (f"model: {model_path}")
    
    # Saves the one_hot_encoder in a file
    one_hot_encoding_path = MODEL_DIR + fn_one_hot_encoding_pickle
    save_model (onehot_encoder, one_hot_encoding_path)
    print (f"one hot encoder: {one_hot_encoding_path}\n\n\n")


if __name__ == "__main__":
  main()

