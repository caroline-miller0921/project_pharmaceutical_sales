# for presentation purposes
import warnings
warnings.filterwarnings("ignore")

import wrangle as w
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from math import sqrt 
from statsmodels.tsa.api import Holt, ExponentialSmoothing

mental_health_df, population_df, rates_df, depressive_rates_df = w.separate_data()

mental_health_df = w.clean_mental_health_data(mental_health_df)
depressive_rates_df = w.clean_depressive_rates_data(depressive_rates_df)
population_df = w.clean_population_data(population_df)
rates_df = w.clean_rates_data(rates_df)

yearly_disorders, yearly_prevalence_sex, yearly_depressive_rates, yearly_suicide_rates = w.yearly_aggregation()

df = w.merge_yearly_aggregation()

train_yearly, test_yearly = w.split_yearly_data()

train, validate, test = w.split_data(df)

train_scaled, validate_scaled, test_scaled = w.scale_data(train, validate, test, columns_to_scale=train.columns)

metric_df = {}

eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])

X_cols = ['prevelance_depressive_disorder', 'anxiety_disorders']

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

def make_predictions(depression=None, anxiety=None):
    yhat_df = pd.DataFrame({'prevelance_depressive_disorder': [depression],
                           'anxiety_disorders': [anxiety]},
                          index=validate.index)
    return yhat_df

def X_y_datasets():

    X_train = train_scaled.drop(columns='prevelance_depressive_disorder')
    X_validate = validate_scaled.drop(columns='prevelance_depressive_disorder')
    X_test = test_scaled.drop(columns='prevelance_depressive_disorder')

    y_train = train['prevelance_depressive_disorder']
    y_validate = validate['prevelance_depressive_disorder']
    y_test = test['prevelance_depressive_disorder']

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def viz_model_performance(y_validate):

    # Visualization of each model's validate performance compared to the baseline and to each other

    plt.figure(figsize=(16,8))
    #actual vs mean
    plt.plot(y_validate.prevelance_depressive_disorder, y_validate.baseline_mean, alpha=.5, color="gray", label='Baseline')

    #actual vs. actual
    plt.plot(y_validate.prevelance_depressive_disorder, y_validate.prevelance_depressive_disorder, alpha=.5, color="red", label='Actual')

    #actual vs. LinearReg model
    plt.scatter(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_lm, 
                alpha=.5, color="indigo", s=100, label="Model: LinearRegression")
    #actual vs. LassoLars model
    plt.scatter(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_lars, 
                alpha=.5, color="purple", s=100, label="Model: Lasso Lars")
    #actual vs. Tweedie/GenLinModel
    plt.scatter(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_glm, 
                alpha=.5, color="violet", s=100, label="Model: TweedieRegressor")
    #actual vs. PolynomReg/Quadratic
    plt.scatter(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_lm2, 
                alpha=.5, color="magenta", s=100, label="Model 2nd degree Polynomial")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("Actual Prevalance of Depression (mil)")
    plt.ylabel("Predicted Prevelence of Depression (mil)")
    plt.title("Model Predictions vs Baseline")
    plt.show()

def OLS(X_train, y_train, X_validate, y_validate):

    # MAKE THE THING: create the model object
    lm = LinearRegression()

    #1. FIT THE THING: fit the model to training data
    OLSmodel = lm.fit(X_train, y_train.prevelance_depressive_disorder)

    #2. USE THE THING: make a prediction
    y_train['depression_pred_lm'] = lm.predict(X_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.prevelance_depressive_disorder, y_train.depression_pred_lm)**.5

    # predict validate
    y_validate['depression_pred_lm'] = lm.predict(X_validate)

    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_lm)**.5

    global metric_df
    metric_df = pd.DataFrame(metric_df)
    metric_df = metric_df.append(
    {
        'model': 'OLS_Regressor',
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        'R2_validate': explained_variance_score(y_validate.prevelance_depressive_disorder, 
                                                y_validate.depression_pred_lm)
    }, ignore_index=True

    )

def Lars(X_train, y_train, X_validate, y_validate):

    # MAKE THE THING: create the model object
    lars = LassoLars(alpha=30.0)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    lars.fit(X_train, y_train.prevelance_depressive_disorder)

    #2. USE THE THING: make a prediction
    y_train['depression_pred_lars'] = lars.predict(X_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.prevelance_depressive_disorder, y_train.depression_pred_lars) ** .5

    #4. REPEAT STEPS 2-3

    # predict validate
    y_validate['depression_pred_lars'] = lars.predict(X_validate)

    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_lars) ** .5

    global metric_df
    metric_df = metric_df.append(
        {
            'model': 'lasso_alpha',
            'RMSE_train': rmse_train,
            'RMSE_validate': rmse_validate,
            'R2_validate': explained_variance_score(y_validate.prevelance_depressive_disorder, 
                                                    y_validate.depression_pred_lars)
        }, ignore_index=True

    )

def GLM(X_train, y_train, X_validate, y_validate):

    # MAKE THE THING: create the model object
    glm = TweedieRegressor(power=1, alpha=500_000)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    glm.fit(X_train, y_train.prevelance_depressive_disorder)

    #2. USE THE THING: make a prediction
    y_train['depression_pred_glm'] = glm.predict(X_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.prevelance_depressive_disorder, y_train.depression_pred_glm)**(1/2)

    #4. REPEAT STEPS 2-3

    # predict validate
    y_validate['depression_pred_glm'] = glm.predict(X_validate)

    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_glm)**(1/2)

    #Append
    global metric_df
    metric_df = metric_df.append(
        {
            'model': 'glm_gamma',
            'RMSE_train': rmse_train,
            'RMSE_validate': rmse_validate,
            'R2_validate': explained_variance_score(y_validate.prevelance_depressive_disorder, 
                                                    y_validate.depression_pred_glm)
        }, ignore_index=True

    )

def PR(X_train, y_train, X_validate, y_validate, X_test):

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.prevelance_depressive_disorder)

    # predict train
    y_train['depression_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.prevelance_depressive_disorder, y_train.depression_pred_lm2)**(1/2)

    # predict validate
    y_validate['depression_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.prevelance_depressive_disorder, y_validate.depression_pred_lm2)**(1/2)

    #Append
    global metric_df
    metric_df = metric_df.append(
        {
            'model': 'quadratic',
            'RMSE_train': rmse_train,
            'RMSE_validate': rmse_validate,
            'R2_validate': explained_variance_score(y_validate.prevelance_depressive_disorder, 
                                                    y_validate.depression_pred_lm2)
        }, ignore_index=True

    )

def establish_baseline(y_train, y_validate):
    
    y_train['baseline_mean'] = y_train.prevelance_depressive_disorder.mean()
    y_validate['baseline_mean'] = y_train.prevelance_depressive_disorder.mean()

    # RMSE of mean
    rmse_train_mu = mean_squared_error(y_train.prevelance_depressive_disorder,
                                    y_train.baseline_mean) ** .5

    rmse_validate_mu = mean_squared_error(y_validate.prevelance_depressive_disorder, 
                                        y_validate.baseline_mean) ** (0.5)

    #Append
    global metric_df
    metric_df = metric_df.append(
        {
            'model': 'mean_baseline',
            'RMSE_train': rmse_train_mu,
            'RMSE_validate': rmse_validate_mu,
            'R2_validate': explained_variance_score(y_validate.prevelance_depressive_disorder,
                                                y_validate.baseline_mean)
        }, ignore_index=True

    )

def get_regression_eval_df(X_train, y_train, X_validate, y_validate, X_test):

    OLS(X_train, y_train, X_validate, y_validate)
    Lars(X_train, y_train, X_validate, y_validate)
    GLM(X_train, y_train, X_validate, y_validate)
    PR(X_train, y_train, X_validate, y_validate, X_test)
    establish_baseline(y_train, y_validate)

    viz_model_performance(y_validate)

    return metric_df

def test_on_best(X_train, y_train, y_validate, X_test, y_test):
    
    # creating Lars object to call
    lars = LassoLars(alpha=30.0)
    lars.fit(X_train, y_train.prevelance_depressive_disorder)

    # creatine baseline object to call
    y_train['baseline_mean'] = y_train.prevelance_depressive_disorder.mean()
    y_validate['baseline_mean'] = y_train.prevelance_depressive_disorder.mean()

    # RMSE of mean
    rmse_train_mu = mean_squared_error(y_train.prevelance_depressive_disorder,
                                    y_train.baseline_mean) ** .5

    # predict on test
    y_test['depression_pred_lars'] = lars.predict(X_test)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.prevelance_depressive_disorder, y_test.depression_pred_lars)**(1/2)

    print("RMSE for LassoLars Model using Linear Regression\nOut-of-Sample Performance: ", rmse_test)
    print()
    # Evaluating the GLM model against the baseline

    print(f'The LassoLars model performed {((rmse_train_mu - rmse_test)/rmse_train_mu) * 100}% better than the baseline')

def last_observed_value():

    last_depressive_rate = train.iloc[-1]['prevelance_depressive_disorder']
    last_anxiety_rate = train.iloc[-1]['anxiety_disorders']
    yhat_df = pd.DataFrame(

        {'prevelance_depressive_disorder': [last_depressive_rate],
        'anxiety_disorders': [last_anxiety_rate]
        }, index=validate.index

    )

    global eval_df
    for col in train[X_cols].columns:
        d = append_eval_df(yhat_df, model_type = 'last_observed_value', target_var = col)
        eval_df = eval_df.append(d, ignore_index=True)

def simple_average():

    avg_depression = train['prevelance_depressive_disorder'].mean()
    avg_anxiety = train['anxiety_disorders'].mean()

    yhat_df = make_predictions(avg_depression, avg_anxiety)

    global eval_df
    for col in train[X_cols].columns:
        d = append_eval_df(yhat_df, model_type = 'simple_average', target_var = col)
        eval_df = eval_df.append(d, ignore_index=True)

def moving_average():

    period = 1 

    rolling_depression = train['prevelance_depressive_disorder'].rolling(period).mean()
    rolling_anxiety = train['anxiety_disorders'].rolling(period).mean()
    rolling_anxiety = rolling_anxiety.tolist()[-1]
    rolling_depression= rolling_depression.tolist()[-1]

    yhat_df = make_predictions(rolling_depression, rolling_anxiety)

    global eval_df
    for col in train[X_cols].columns:
        d = append_eval_df(yhat_df, model_type = 'moving_average', target_var = col)
        eval_df = eval_df.append(d, ignore_index=True)

def holts_linear_trend():

    yhat_df = {}
    yhat_df = pd.DataFrame(yhat_df)

    # doing this in a loop for each column
    for col in train[X_cols].columns:
        model = Holt(train[col], exponential=False, damped=True)
        model = model.fit(optimized=True)
        yhat_items = model.predict(start = validate.index[0],
                                end = validate.index[-1])
        yhat_df[col] = yhat_items

    global eval_df
    for col in train[X_cols].columns:
        d = append_eval_df(yhat_df, model_type = 'holts_optimized', target_var = col)
        eval_df = eval_df.append(d, ignore_index=True)

def get_time_series_eval_df():

    last_observed_value()
    simple_average()
    moving_average()
    holts_linear_trend()

    return eval_df

def test_best_forecaster():

    period = 1 

    rolling_depression = train['prevelance_depressive_disorder'].rolling(period).mean()
    rolling_anxiety = train['anxiety_disorders'].rolling(period).mean()
    rolling_anxiety = rolling_anxiety.tolist()[-1]
    rolling_depression= rolling_depression.tolist()[-1]

    yhat_df = pd.DataFrame(
    {
        'prevelance_depressive_disorder': [rolling_depression],
         'anxiety_disorders': [rolling_anxiety]},
                          index=test.index)
    
    rmse_depression = sqrt(mean_squared_error(test['prevelance_depressive_disorder'], 
                                       yhat_df['prevelance_depressive_disorder']))

    rmse_anxiety = sqrt(mean_squared_error(test['anxiety_disorders'], 
                                        yhat_df['anxiety_disorders']))
    
    print(f'The RMSE on the test dataset for depression is {rmse_depression} using the rolling average model.')
    print(f'The RMSE on the test dataset for anxiety is {rmse_anxiety} using the rolling average model.')