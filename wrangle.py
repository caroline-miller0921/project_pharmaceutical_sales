from lib2to3.pgen2.pgen import DFAState
import os
import numpy as np
import pandas as pd
from datetime import datetime

def get_mental_health_data():

    '''
    Function which loads the csv file if saved locally. If csv file
    is not saved in same repositiory, print statement explains to
    go to Kaggle and download the csv. 
    https://www.kaggle.com/datasets/thedevastator/global-mental-health-disorders 
    '''

    if os.path.isfile('mental_health_data.csv'):
        df = pd.read_csv('mental_health_data.csv')
        return df
    else:
        print('Please save the .csv file locally from Kaggle.')

def separate_data():

    ''''
    Function which separates the dataframe into the four included datasets
    '''

    df = get_mental_health_data()

    # renaming columns
    df = df.rename(columns={
        'Entity': 'entity',
        'Year': 'year', 
        'Code': 'code', 
        'Schizophrenia (%)': 'schizophrenia',
        'Bipolar disorder (%)': 'bipolar_disorder',
        'Eating disorders (%)': 'eating_disorders',
        'Anxiety disorders (%)': 'anxiety_disorders',
        'Drug use disorders (%)': 'drug_use_disorders',
        'Depression (%)': 'depression',
        'Alcohol use disorders (%)': 'alcohol_use_disorders'})
    
    mental_health_df = df[:6468]
    population_df = df[6469:54276]
    rates_df = df[54277:102084]
    depressive_rates_df = df[102085:]

    return mental_health_df, population_df, rates_df, depressive_rates_df

def clean_mental_health_data(mental_health_df):

    '''
    Takes in the first dataframe parsed from the larger dataset as a parameter.
    Converts all numeric data to float datatypes. Converts year to datetime.
    Fills nulls in code with the entity name. Sets the index to the year.    
    '''

    df = mental_health_df

    for col in df.columns:
        print(f'There are {df[col].isna().sum()}, {round(((df[col].isna().sum()) / len(df) * 100), 2)}%, null values in {col}')

    df['schizophrenia'] = df.schizophrenia.astype(float)
    df['bipolar_disorder'] = df.bipolar_disorder.astype(float)
    df['eating_disorders'] = df.eating_disorders.astype(float)

    df['year'] = df.year.astype(int)

    df['code'] = df['code'].fillna(df['entity'])

    df = df.set_index('year')

    df.drop(columns='index', inplace=True)

    return df

def clean_population_data(population_df):

    df = population_df

    for col in df.columns:
        print(f'There are {df[col].isna().sum()}, {round(((df[col].isna().sum()) / len(df) * 100), 2)}%, null values in {col}')

    df = df.drop(columns={'anxiety_disorders',
                          'drug_use_disorders',
                          'depression',
                          'alcohol_use_disorders'})
    
    df = df.rename(columns={'schizophrenia': 'prevalence_males', 
                            'bipolar_disorder': 'prevalance_female',
                            'eating_disorders': 'effected_population'})
    
    df['year'] = df['year'].str.replace(' BCE', '')

    df['year'] = df.year.astype(int)

    df = df[(df['year'] >= 1990) & (df['year'] <= 2017)]

    df = df.dropna()

    df['code'] = df['code'].fillna(df['entity'])

    df['prevalence_males'] = df.prevalence_males.astype(float)
    df['prevalance_female'] = df.prevalance_female.astype(float)
    df['effected_population'] = df.effected_population.astype(float)

    df = df.set_index('year')

    df.drop(columns='index', inplace=True)

    return df

def clean_rates_data(rates_df):

    rates_df = rates_df.drop(columns={'anxiety_disorders', 
                                      'drug_use_disorders', 
                                      'depression', 
                                      'alcohol_use_disorders'})
    
    rates_df = rates_df.rename(columns={'schizophrenia': 'suicide_rates_per_100k', 
                                        'bipolar_disorder': 'depressive_disorder_rates_per_100k', 
                                        'eating_disorders': 'population'})
    
    rates_df['year'] = rates_df.year.str.replace(' BCE', '')

    rates_df['year'] = rates_df.year.astype(int)

    rates_df = rates_df[(rates_df['year'] >= 1990) & (rates_df['year'] <= 2017)]

    rates_df = rates_df.set_index('year')

    rates_df['suicide_rates_per_100k'] = rates_df.suicide_rates_per_100k.astype(float)
    rates_df['depressive_disorder_rates_per_100k'] = rates_df.depressive_disorder_rates_per_100k.astype(float)
    rates_df['population'] = rates_df.population.astype(float)

    for col in rates_df.columns:
        print(f'There are {rates_df[col].isna().sum()}, {round(((rates_df[col].isna().sum()) / len(rates_df) * 100), 2)}%, null values in {col}')

    rates_df = rates_df.dropna()

    rates_df['percentage_suicide'] = ((rates_df.suicide_rates_per_100k / 100_000) * 100)
    rates_df['percentage_depressive_disorder'] = ((rates_df.depressive_disorder_rates_per_100k / 100_000) * 100)
    rates_df['num_suicide'] = round(rates_df.percentage_suicide * rates_df.population)
    rates_df['num_depressed'] = round(rates_df.percentage_depressive_disorder * rates_df.population)

    rates_df.drop(columns='index', inplace=True)

    return rates_df

def clean_depressive_rates_data(depressive_rates_df):

    depressive_rates_df = depressive_rates_df.drop(columns={'bipolar_disorder',
       'eating_disorders', 'anxiety_disorders', 'drug_use_disorders',
       'depression', 'alcohol_use_disorders'})
    
    depressive_rates_df = depressive_rates_df.rename(columns={'schizophrenia': 'prevelance_depressive_disorder'})
    
    depressive_rates_df['year'] = depressive_rates_df.year.astype(int)

    depressive_rates_df = depressive_rates_df.set_index('year')

    depressive_rates_df['prevelance_depressive_disorder'] = depressive_rates_df.prevelance_depressive_disorder.astype(float)

    depressive_rates_df.drop(columns='index', inplace=True)

    depressive_rates_df['code'] = depressive_rates_df['code'].fillna(depressive_rates_df['entity'])

    return depressive_rates_df

def yearly_aggregation():

    mental_health_df, population_df, rates_df, depressive_rates_df = separate_data()

    mental_health_df = clean_mental_health_data(mental_health_df)
    population_df = clean_population_data(population_df)
    depressive_rates_df = clean_depressive_rates_data(depressive_rates_df)
    rates_df = clean_rates_data(rates_df)

    yearly_disorders = mental_health_df.groupby('year').mean()
    yearly_prevalence_sex = population_df.groupby('year').sum()
    yearly_depressive_rates = depressive_rates_df.groupby('year').mean()
    yearly_suicide_rates = rates_df.groupby('year').mean()

    return yearly_disorders, yearly_prevalence_sex, yearly_depressive_rates, yearly_suicide_rates

def merge_yearly_aggregation():

    df_1, df_2, df_3, df_4 = yearly_aggregation()

    df = df_1.join(df_2)

    df = df.join(df_3)

    df = df.join(df_4)

    return df

