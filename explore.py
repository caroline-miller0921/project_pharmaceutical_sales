# transform
from cProfile import label
import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter


def create_diff_df(mental_health_df):

    diff_df = mental_health_df.copy()
    for col in diff_df.columns:
        if col != 'entity' and col != 'code':
            diff_df[f'last_year_diff_{col}'] = diff_df[col] - diff_df[col].shift(1)

    diff_df.drop(columns={'schizophrenia', 'bipolar_disorder',
       'eating_disorders', 'anxiety_disorders', 'drug_use_disorders',
       'depression', 'alcohol_use_disorders'}, inplace=True)
    diff_df = diff_df.groupby('year').mean()

    return diff_df

def viz_diff_df(diff_df):

    for col in diff_df.columns:
        if col != 'entity' and col != 'code':
            diff_df[col][1:].plot()
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.title('Difference in World Rates of Mental Health')
            plt.xlabel('Year')
            plt.ylabel('Diference in Rate of Prevelence')

def start_to_end_diff(rates_df, yearly_disorders):

    changes_disorders = pd.concat([yearly_disorders.head(1), yearly_disorders.tail(1)])
    changes_disorders = changes_disorders.T

    changes_disorders[f'change_over_time'] = changes_disorders[2017] - changes_disorders[1990]

    change_in_rates = rates_df.copy()
    change_in_rates = change_in_rates.groupby('year').mean()
    change_in_rates = pd.concat([change_in_rates.head(1), change_in_rates.tail(1)])
    change_in_rates = change_in_rates.T
    change_in_rates['change_over_time'] = change_in_rates[2017] - change_in_rates[1990]

    return change_in_rates

def viz_us_vs_world(us_rates_df, world_rates_df):

    # plot to visualize actual vs predicted models
    fig, (ax0, ax1) = plt.subplots(1, 2)

    plt.figure(figsize=(5, 3))

    ax0.hist(us_rates_df.percentage_depressive_disorder, color='violet', alpha=.5, edgecolor='black')
    ax1.hist(world_rates_df.percentage_depressive_disorder, color='indigo', alpha=.5, edgecolor='black')

    ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)

    ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) 
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) 

    ax0.set_title("US Prevalence of Depressive Disorders (%)")
    ax1.set_title("Global Prevalence of Depressive Disorders (%)")
    fig.suptitle("Comparing the Distribution of Depression Prevalence of the World vs the US")
    fig.tight_layout()
    plt.show()

def viz_happy_unhappy(afghanistan, norway):

    # plot to visualize actual vs predicted models
    fig, (ax0, ax1) = plt.subplots(1, 2)

    plt.figure(figsize=(5, 3))

    ax0.hist(afghanistan, color='violet', alpha=.5, edgecolor='black')
    ax1.hist(norway, color='indigo', alpha=.5, edgecolor='black')

    ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)

    ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) 
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) 

    ax0.set_title("Afghanistan's Prevalence of Depressive Disorders (%)")
    ax1.set_title("Norway's Prevalence of Depressive Disorders (%)")
    fig.suptitle("Distribution of Depression Prevalence of the Happiest vs the Unhappiest Countries")
    fig.tight_layout()
    plt.show()

def viz_happy_vs_world(happy_countries, world):

    # plot to visualize actual vs predicted models
    fig, (ax0, ax1) = plt.subplots(1, 2)

    plt.figure(figsize=(5, 3))

    ax0.hist(happy_countries, color='violet', alpha=.5, edgecolor='black')
    ax1.hist(world, color='indigo', alpha=.5, edgecolor='black')

    ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)

    ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) 
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) 

    ax0.set_title("Top 5 Happiest Countries'\n Prevalence of Depressive Disorders (%)")
    ax1.set_title("Global Prevalence of Depressive Disorders (%)")
    fig.suptitle("Distribution of Depression Prevalence of the Happiest Countries vs the World")
    fig.tight_layout()
    plt.show()

def viz_gender_diff(population_df):

    population_yearly = population_df.groupby('year').mean()

    plt.plot(population_yearly.index, population_yearly.prevalence_males, color='indigo', label='Males')
    plt.plot(population_yearly.index, population_yearly.prevalance_female, color='violet', label='Females')
    plt.ylim(ymax = 6, ymin = 0)
    plt.title('Trend of Mental Health Between Males and Females')
    plt.xlabel('Year')
    plt.ylabel('Number of People (%)')
    plt.legend()
    plt.show()

