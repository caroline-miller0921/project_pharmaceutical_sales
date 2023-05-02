Project Plan

1. Acquire:
  - Search for an intriguing dataset for which Time Series modeling can be conducted.
2. Prepare/Wrangle:
  - remove outliers
  - correct datatypes
  - scale data
  - create dummies for categorical features
  - deal with null values
  - split the data
3. Explore
  - create univariate and bivariate visulations
  - determine questions 
  - visulaize and use statistical data to attempt to answer the questions
4. Modeling
  - conduct time series modeling to forecast depression rates 
  - conduct linear regression modeling
  - evaluate model preformance
  - evaluate best performing model on test dataset
5. Conclude
  - explain the key findings
  - make recommendations
  - establish a way forward

Project Description:

This project looks at meantal health data from every country spanning 1990 to 2017. Global spikes in mental health disorders affects me
personally, and the effects of living with these disorders can be devasting to both the afflicted individual and their friends and families.
The project aims to detect trends in these disorders, explore the different distributions throughout the world, and predict how these
disorders may affect the citizens of the world in the future. the project uses time series forecasting and linear regression to predict
depression on a global scale.

Project Goal:

The endstate after conducting this project is a model which can predict depression rates better than the baseline model and to provide some
conclusions as to why these disorders are affecting the global population and which regions of the world are most afflicted.

Executive Summary:

  The original intent for this project was to cerate a model which could effectively predict the trends for mental health disorders,
  create features based off of the populations of the countries who consume the most pharmaceutical products using the prevalence rates, 
  and to try and calculate the annual sales of these drugs.
  
  This project currently meets the first goal outlined. the LassoLars model performs 87% better than the mean baseline for depression
  rates. 
  
  Key findings:
  
    The top 5 "happiest countries" according to the World Happines report from 2017 actually have higher depression rates than the global
    mean depression rate. 
    
    The average depression and anxiety rates for the least happy nation is higher than the happiest nation.
    
    The anxiety and depression rates in the US are higher than at of the world. 
    
    Prevalence of all disorders besides depression are increasing. 

Data Dictionary

1. Entity	The name of the country or region. (String)
2. Code	The ISO code of the country or region. (String)
3. Year	The year the data was collected. (Integer)
4. Schizophrenia (%)	The percentage of people with schizophrenia in the country or region. (Float)
5. Bipolar disorder (%)	The percentage of people with bipolar disorder in the country or region. (Float)
6. Eating disorders (%)	The percentage of people with eating disorders in the country or region. (Float)
7. Anxiety disorders (%)	The percentage of people with anxiety disorders in the country or region. (Float)
8. Drug use disorders (%)	The percentage of people with drug use disorders in the country or region. (Float)
9. Depression (%)	The percentage of people with depression in the country or region. (Float)
10. Alcohol use disorders (%)	The percentage of people with alcohol use disorders in the country or region. (Float)

Steps to Reproduce:

1. Download and save CSV from the kaggle link in the report locally in the same repo you wish to conduct your work
2. Split the dataframe into the four distinct datasets
3. Wrangle each dataset by correcting datatypes, dropping erroneous columns, rename columns to match the dataset
4. Conduct exploratory analysis using statistical testing, analyzing subsets, and making visualizations
5. Scale and split the data
6. Create unique models to predict your target variable
7. Evaluate models
8. Evaluate test dataset on best performing model
9. Draw conclusions

Questions to Explore:

1. Is there a difference in the rate of depression in the United States versus the rest of the world?
2. Are the rates of any of the mental health disorders decreasing? What is the rate for each disorder from 1990 to 2017?
3. Is there a difference in the rates of the happiest country vs the least happy country?
4. Is there a difference in rates of depression between two populations: 1) Iceland, Portugal, Canada, Australia and Sweden and 2) the rest of the world?
