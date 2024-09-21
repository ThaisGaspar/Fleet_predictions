This code predicts the expenses of a fuel carrier fleet based on historical data from multiple sources such as vehicle information, revenue records, fleet expenses, and kilometers driven. Below is a guide to understanding and using the code:

Key Files Required
frota.csv: Contains vehicle details such as type, brand, model, and capacity.
faturamento_2020_limpos.csv: Revenue data for the fleet in 2020.
gastos_frota.csv & gastos_frota_2021.csv: Fleet expenses for 2020 and 2021.
gastos_km_limpo.csv: Data on kilometers driven and fuel consumption.
Output files: richfleet.csv, richKMexpenses.csv, richrevenue.csv are generated and used for further analysis.
Steps in the Code
Data Collection & Preparation

Reads CSV files into Pandas DataFrames.
Cleans column names, removes extra spaces, and standardizes text formats.
Corrects missing and erroneous values, especially in numerical fields like capacity and total expenses.
Data Enrichment

Merges fleet, revenue, and expense datasets to create a richer, unified dataset.
Applies text transformations and removes unnecessary spaces for standardization.
Exploratory Data Analysis (EDA)

Descriptive statistics and checks for inconsistencies in columns like capacity, quantity, model_year.
Time series analysis for expenses and revenues over months/years.
Generates visualizations such as line plots for expenses over time and pie charts to analyze departmental expenses.
Data Cleaning & Outlier Handling

Identifies and removes outliers using box plots and histograms.
Uses median and most frequent values to impute missing data.
Clustering & Segmentation

Implements KMeans clustering to segment vehicles or expenses into distinct groups.
Uses PCA to reduce dimensions and visualize clusters.
Provides detailed analysis for each cluster, identifying common characteristics and behaviors.
Predictive Modeling

Several regression models (Linear, Ridge, Decision Tree) are applied to predict fleet expenses.
Uses both simple linear regression and multiple linear regression models.
Applies log transformation to certain variables to improve model performance.
