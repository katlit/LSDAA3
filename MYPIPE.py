#MYPIPE

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt


import getData_Prepros

import mlflow
import mlflow.sklearn


x_train, y_train, x_val, y_val, x_test, y_test = getData_Prepos.get_train_test("dataset.json")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def get_model(model, params):
    pipeline = Pipeline([
        ("WindTransformer", Convert_direction()),
        ("Standard Scaler", StandardScaler()),
        ("Model", model)
    ])
    clf = GridSearchCV(pipeline, params)
    clf.fit(x_train, y_train)
   
    return clf

#WindTransformer(): {"Windtransform_how": ["one-hot", "speed-vector", "vectorize"]},

model_dict = {
    KNeighborsRegressor(): {"Model__n_neighbors": [5, 6, 7, 8, 9, 10]},
    DecisionTreeRegressor(): {"Model__max_depth": [None, 5, 6, 7, 8],
                             "Model__min_samples_leaf": [1, 2, 4, 5, 6, 7, 8]},
    SVR(): {"Model__kernel": ["rbf", "sigmoid", "poly"],
            "Model__C": [ 2.0, 1.5, 1.0, 0.5],
           "Model__gamma": ['scale', 'auto']},
    SGDRegressor(): {"Model__alpha": [0.1, 0.05, 0.01, 0.001, 0.0001],
                    "Model__alpha": [0.0001, 0.001],
                    "Model__learning_rate": ["invscaling", "optimal", "adaptive"]}
}


def find_best_model():
    """
    Runs "get_model" and saves the best model & parameters from model_dict to disk
    """
    MSE = 1000 #The MAE is initialised high, to ensure that a model is saved
    best_model = None

    for model, params in model_dict.items(): #Loop through models and given hyperparameters
        clf = get_model(model, params) #Call the 'pipeline function'
        y_hat = clf.predict(x_val) #predict on the test data w. given model and "best" parameters
        temp_mse = mean_squared_error(y_val, y_hat)

        ###############MLFLOW LOG##########################################
        paramlist = [(k,v) for k, v in clf.best_params_.items()]
        for i in range(len(paramlist)):
            param_name, param_value = paramlist[i]
            mlflow.log_param(param_name, param_value)

        print("Best hyperparameters for model: {}\nLowest MSE: {}\n-------------------------".format(
            clf.best_estimator_[-1], temp_mse)) #Print info about the given model
        if temp_mse < MSE:
            MSE = temp_mse
            best_model = clf.best_estimator_ #Save the model if it performs better on the test data than previous best
    
    mlflow.sklearn.log_model(best_model, "best_model")
    joblib.dump(best_model, 'BestModel.pkl') #Save model
    print("\nModel Saved")
    return best_model

with mlflow.start_run():
    best_m = find_best_model()


