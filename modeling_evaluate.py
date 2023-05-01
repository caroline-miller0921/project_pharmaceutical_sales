import wrangle as w
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt 

mental_health_df, population_df, rates_df, depressive_rates_df = w.separate_data()

mental_health_df = w.clean_mental_health_data(mental_health_df)
depressive_rates_df = w.clean_depressive_rates_data(depressive_rates_df)
population_df = w.clean_population_data(population_df)
rates_df = w.clean_rates_data(rates_df)

yearly_disorders, yearly_prevalence_sex, yearly_depressive_rates, yearly_suicide_rates = w.yearly_aggregation()

df = w.merge_yearly_aggregation()

train_yearly, test_yearly = w.split_yearly_data()

train, validate, test = w.split_data(df)

def evaluate(yhat_df, target_var):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse

def plot_and_eval(yhat_df, target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(yhat_df, target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

# eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
# eval_df

# function to store the rmse so that we can compare
def append_eval_df(yhat_df, model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(yhat_df, target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return d