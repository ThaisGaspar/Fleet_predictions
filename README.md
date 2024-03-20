# Fleet_predictions

This code is to predict the expenses of a fuel carrier fleet

#**Collect Data**

*   Padronize columns names
*   Merge the data in vehicle information(frota.csv)

*   Standardize textual values
import os
import numpy as np
import pandas as pd
import re

##Fleet

df_fleet = pd.read_csv('frota.csv')
df_fleet.head()

df_fleet.columns = map(str.lower,df_fleet.columns)

#changing columns names
df_fleet.rename(columns={'tipo veiculos':'vehicle_type', 'tipo carroce':'truck_type',
                         'marca':'brand_name','modelo':'model', 'tipo veículo': 'vehicle_type',
                         'capacidade':'capacity', 'depto.': 'department','placa':'license',
                         'ano/modelo': 'model_year','coligada':'afiliate_name',
                         'aplicação':'aplication', 'proprietário':'property_owner'},inplace=True)

df_fleet.info()

len(df_fleet['license'].unique())

df_fleet.shape

df_fleet['afiliate_name'].unique()

df_fleet['brand_name'].unique()

##**Revenues 2020**

df_revenues20 = pd.read_csv("faturamento_2020_limpos.csv")
df_revenues20.columns = map(str.lower,df_revenues20.columns)

#changing columns names
df_revenues20.rename(columns={'codigo_filial':'branch_name','nf__placa':'license',
                              'grupo_produto__codigo':'product_code','grupo_produto__descricao': 'product_group_descpription', '1item_nf_0':'iv_item_10',
                              'grade total':'total'},inplace=True)

df_revenues20.head()

df_revenues20['license'] = df_revenues20['license'].str.replace(' ','')

df_revenues20['product_group_descpription'].unique()

##**Fleet expenses**

#fleet spends
df_fleetSpend20 = pd.read_csv('gastos_frota.csv')
df_fleetSpend20.shape

df_fleetSpend20.head(2)

#changing columns names
df_fleetSpend20.rename(columns={'NOME_FILIAL':'branch_name','VEICULO_PLACA':'license',
                              'PLANO_CONTA__DESCRICAO':'area_description','DATA':'date','DOCUMENTO':'document',
                              'CENTRO_CUSTO__DESCRICAO':'tax_description','PRODUTO_DESCRICAO':'product_description',
                              'MOVIMENTO__FORNECEDOR':'supplier','CENTRO_CUSTO__CODIGO':'tax_code',
                              'QUANTIDADE':'quantity','TOTAL':'total'},inplace=True)

df_fleetSpend20.head()

##**Fleet Spends 2021**

df_fleetSpends21 = pd.read_csv('gastos_frota_2021.csv')
df_fleetSpends21.shape

df_fleetSpends21.rename(columns={'NOME_FILIAL':'branch_name', 'VEICULO_PLACA':'license',
                              'PLANO_CONTA__DESCRICAO':'area_description','DATA':'date','DOCUMENTO':'document',
                              'CENTRO_CUSTO__DESCRICAO':'tax_description','PRODUTO_DESCRICAO':'product_description',
                              'MOVIMENTO__FORNECEDOR':'supplier','CENTRO_CUSTO__CODIGO':'tax_code',
                              'QUANTIDADE':'quantity','TOTAL':'total'},inplace=True)

#delete lines with word total
regex=".*Total"
df_fleetSpends21=df_fleetSpends21[~df_fleetSpends21["license"].str.contains(regex)]
df_fleetSpends21=df_fleetSpends21[~df_fleetSpends21["branch_name"].str.contains(regex)]
df_fleetSpends21.shape

##**Concatenate the databases**

#concatenating the two expense bases
df_fleetSpend = df_fleetSpend20.append(df_fleetSpends21)
df_fleetSpend.head(2)
df_fleetSpend.shape

##**KM Expenses**

df_expensesKM = pd.read_csv('gastos_km_limpo.csv')
df_expensesKM.columns=map(str.lower,df_expensesKM.columns)
df_expensesKM.rename(columns={'data':'date','placa':'license','tipo':'type','fornecedor':'supplier',
                             'litros':'liters'},inplace=True)
df_expensesKM.head(2)

#removing the spaces from the license to standardize like the other tables,
#facilitating the crossing of information
df_fleet['license']=df_fleet['license'].str.replace(' ','')
df_fleetSpend['license']=df_fleetSpend['license'].str.replace(' ','')

df_richfleet = df_fleetSpend.merge(df_fleet, how = 'left',left_on=['license'],right_on=['license'])
df_richfleet.head(3)

df_richfleet.duplicated().sum()
df_richfleet.drop_duplicates(inplace=True)
df_richfleet.shape

**Enriching the database KM Expenses**

df_richkmexpenses = df_expensesKM.merge(df_fleet, how='left', left_on=['license'], right_on=['license'])
df_richkmexpenses.head(3)

**Enriching Revenues Database**

df_richrevenue = df_revenues20.merge(df_fleet, how='left', left_on=['license'], right_on=['license'])
df_richrevenue.head(3)

df_richrevenue['branch_name'].unique()
df_richfleet['branch_name'].unique()

df_richrevenue['branch_name'].unique()

df_richrevenue.head(3)

#removing the branch code in the revenue table and leaving only the name.
df_richrevenue['branch_name'] = df_richrevenue['branch_name'].apply(lambda x: re.sub('^[0-9]*\s-\s','', x))

#confirming the change in the branch name of the revenue table
df_richrevenue['branch_name'].unique()

df_richrevenue.head(2)

#defaulting text column values to uppercase and removing extra empty space in all tables
df_richrevenue=df_richrevenue.apply(lambda x: x.astype(str).str.upper().str.strip())

df_richfleet=df_richfleet.apply(lambda x: x.astype(str).str.upper().str.strip())

df_richkmexpenses=df_richkmexpenses.apply(lambda x: x.astype(str).str.upper().str.strip())

df_fleet['afiliate_name'].unique()

#**Saving Data**
df_richfleet.to_csv('richfleet.csv',index=False)
df_richkmexpenses.to_csv('richKMexpenses.csv',index=False)
df_richrevenue.to_csv('richrevenue.csv', index=False)

#**Data Exploration**

## Loading the csv files
# setting NAN text and - as NULL value
df_fleet = pd.read_csv("richfleet.csv", na_values=['NAN', '-'])
df_fleet.info()
df_fleet.shape

**Analyse:**

- As date columns and capacity are as string

- column capacity is with high null value.

df_fleet.columns

#Some Basic Statistics of Numerical Variables in the Table
df_fleet.describe().round(2)

#check the unique values of each column for inconsistencies
for col in df_fleet.columns:
  print("="*15,col,':')
  print(df_fleet[col].unique())
  print('\n')
#SINGLE VALUES ANALYSIS

*   column 'date' is with total values by date
*   column 'tax_description' is with total values by category
*   several variables have a category called 'NAN' (null)
*   'document' column is completely out of standard, some are with acronym+number
*   column 'capacity' should be numeric, but some vehicles that are with the same type: MUNCK, MUCK, CAVALO
*   columns 'capacity', 'total', 'quantity', 'model_year' are string in some or all values.

#removing the word TOTAL from the date, area_description and product_description
regex = "TOTAL"
df_fleet[["date",'area_description','product_description']] = df_fleet[["date",'area_description','product_description']].apply(lambda x: x.str.replace(regex,'') )

## REPLACING MISSING VALUES WITH NAN
df_fleet = df_fleet.replace(r'^\s*$', np.nan, regex=True)

# CORRECTING WRONG VALUES ON CAPACITY COLUMN
mistakes = ['MUNCK', 'MUCK', 'CAVALO']
df_fleet.loc[(df_fleet['capacity'].isin(mistakes)),'capacity'] = 0

#transforming the field to numeric
df_fleet['capacity'] = pd.to_numeric(df_fleet['capacity'])

#create month and year column from the date
df_fleet['date'] = pd.to_datetime(df_fleet.date, dayfirst=True)
df_fleet['year'] = df_fleet['date'].dt.year
df_fleet['month'] = df_fleet['date'].dt.month

#the department's category was duplicated, with one having a space at the end,
#as well as some branch name categories.
df_fleet['department'] = df_fleet['department']
df_fleet['branch_name'] = df_fleet['branch_name'].str.rstrip()
df_fleet['model'] = df_fleet['model'].str.rstrip()

# Null Value Analysis
df_fleet[df_fleet['capacity']>10].head(3)

df_fleet[df_fleet['capacity'].isna()]['model'].value_counts()

#all light utilities have, according to research,
#from ~630 KG to 756 KG, an average of this will be set for all of them
regex = ".*SAVEIRO.*|.*STRADA.*|.*MONTANA.*|.*OROCH.*"
df_fleet.loc[df_fleet['model'].str.contains(regex, na=False), 'capacity'] = 693

#all passengers vehicles were set to 0, as they are not cargo vehicles
regex = ".*GOL.*|.*VOYAGE.*|.*UP.*|.*JETTA.*"
df_fleet.loc[df_fleet['model'].str.contains(regex, na=False), 'capacity'] = 0

#all pickups have a capacity just above 1000KG
regex = ".*F.*350.*|.*F.*1000.*|.*L.*200.*|.*S-10.*|.*AMAROK.*|.*SILVERADO.*|D-20"
df_fleet.loc[df_fleet['model'].str.contains(regex, na=False), 'capacity'] =1100

df_fleet[df_fleet['model'].str.contains(regex, na=False)]['capacity'].isna().sum()

# The isna function checks for null values in the entire table, and then the percentage of these values for each column is calculated.
(df_fleet.isna().sum()/len(df_fleet)*100).sort_values(ascending=False)

Null values analysis

The columns have a reasonably low NULL value, they could be excluded, however, as the number of records is already small, these null values will be corrected (inputed)

# CORRECTION OF NULL VALUES

*   Numerical variables will be corrected by the median
*   Categorical variables will be corrected by the most frequent value

numerical_columns = df_fleet.columns[[i in ['float', 'int'] for i in [df_fleet[c].dtype for c in df_fleet.columns]]]
numerical_columns

textual_columns = df_fleet.columns[ [ i in ['O'] for i in [df_fleet[c].dtype for c in df_fleet.columns] ]]
textual_columns

## CORRECTING NULL VALUES IN NUMERICAL COLUMNS
for c in numerical_columns:
  media = df_fleet[c].median()
  print(c, media)
  df_fleet[c].fillna(media,inplace=True)

## CORRECTING NULL VALUES IN TEXTUAL COLUMNS
for c in textual_columns:
  if c not in ['area_description']:
#COLUMN area_description WILL NOT BE CORRECTED, HIGH NULL VALUE THE IDEAL IS TO REMOVE THIS COLUMN IN THE MODELING PART,
#IT WILL BE MAINTAINED TO CARRY OUT THE DESCRIPTIVE FOLLOWING PART AND THE RESULTS OF THE MODEL
    media = df_fleet[c].value_counts().index[0]
    print(c, media )
    df_fleet[c].fillna(media,inplace=True)

(df_fleet.isna().sum()/len(df_fleet)*100).sort_values(ascending=False)

#**DESCRIPTIVE ANALYSIS**


*   apply data analysis in order to identify useful information, patterns and the like, both for the business (insights) and to model the next step (algorithms)
*   initially apply to the df_richfleet base (if applicable, we replicate analyzes for the other bases)

### Time series analysis
- total cost over time

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

#plot graphics for analysis
plt.figure(figsize=(15,5))
plt.title("Sum of Expenses by Date")
df_fleet.groupby(['date'])['total'].sum().plot(kind='line')
plt.show()

plt.figure(figsize=(15,5))
plt.title("Average of Expenses")
df_fleet.groupby(['year','month'])['total'].mean().plot(kind='line')
plt.show()

#plot expenses average
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot()

plt.title("Average of Expenses by YEAR-MONTH")
df_fleet[df_fleet['year']==2020].groupby(['month'])['total'].mean().plot(kind='line', legend=True, ax=ax)
df_fleet[df_fleet['year']==2021].groupby(['month'])['total'].mean().plot(kind='line', legend=True, ax=ax)
ax.legend(['2020', '2021'])
plt.show()

plt.figure(figsize=(15,7))
plt.title("Total Expenses by department")
total_dpto = df_fleet.groupby(['department','year', 'month'])['total'].sum().reset_index()#.plot(kind='line', legend=True, ax=ax)
total_dpto['date'] = df_fleet.apply(lambda x: pd.to_datetime(f"{int(x['year'])}-{int(x['month'])}-01"),axis=1)
total_dpto
sns.lineplot(x = 'month', y = 'total',data = total_dpto,hue='department',ci= None)
plt.show()

##Comparative analysis with pie chart
Division of services (lines) by department

fig = plt.figure(figsize=(10,8))

plt.title('% Records by Department')
(df_fleet.groupby('department')['department'].count().sort_values()).plot(kind='pie', autopct='%1.1f%%')
plt.show()

fig = plt.figure(figsize=(10,8))

plt.title('% Expenses by Department')
(df_fleet.groupby('department')['total'].sum().sort_values()).plot(kind='pie', autopct='%1.1f%%')
plt.show()

#ANALYSIS OF PROPORTIONS

More than 70% of the records are related to OPERATIONAL, and the volume of expenses corresponds to more than 80% of the total
COMMERCIAL is the second department with the most records in the database, both in number of records and in value spent

#Comparative analysis with bars

*  Frequency (absolute or relative) of services (lines) by the branch name
*  Frequency (absolute or relative) of services (lines) by the license
*   Frequency (absolute or relative) of services (lines) by area description
*   Frequency (absolute or relative) of services (lines) by supllier
*   Frequency (absolute or relative) of services (lines) by tax_description
*   Frequency (absolute or relative) of services (lines) by product
*   Frequency (absolute or relative) of services (lines) by brand name
*   Frequency (absolute or relative) of services (lines) by vehicle type
*   Frequency (absolute or relative) of services (lines) by model year
*   Frequency (absolute or relative) of services (lines) by the affiliate
*   Frequency (absolute or relative) of services (lines) by the owner
*   Frequency (absolute or relative) of services (lines) by the affiliate name
*   Frequency (absolute or relative) of services (lines) by the aplication
*   Frequency (absolute or relative) of services (lines) by the truck type
*   Frequency (absolute or relative) of services (lines) by the department

#Comparative analysis with boxplot

*   Cost variation by afiliate
*   Cost variation by license
*   Cost variation by supplier
*   Cost variation by tax_description
*   Cost variation by product_description
*   Cost variation by brand name
*   Cost variation by vehicle_type
*   Cost variation by model_year
*   Cost variation by branch_name
*   Cost variation by ownership

#BOXPLOT TO VERIFY VARIATION AND OUTLIER OF NUMERICAL VARIABLES
i = 1
fig = plt.figure(figsize = (20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for j,c in enumerate(df_fleet.columns):
    if df_fleet[c].dtypes != 'O' and c not in ['date','year','document','tax_code']:
        ax = fig.add_subplot(5, 4, i)
        sns.boxplot(y = c,data = df_fleet,ax = ax)
        plt.title(c)
        i +=  1

**BOXPLOT ANALYSIS**

It can be observed that there is an outlier that greatly distorts the behaviour of the data in the QUANTITY and TOTAL variables

in the capacity column there are few outliers.

#BOXPLOT OF CATEGORY VARIABLES BY TOTAL VARIABLE
#PURPOSE IS TO IDENTIFY THE BEHAVIOUR OF THE VARIABLES FOR THE (TOTAL) MAINTENANCE EXPENSES,
#IF THERE IS DISTINCTION OR PATTERN IN BEHAVIOUR.
i = 1
fig = plt.figure(figsize = (25,25))
fig.subplots_adjust(hspace=1.5, wspace=0.4)
for j,c in enumerate(df_fleet.columns): #['VL_FLX_VNCT','VL_VNCT','VL_FATM','IN_TRNS_TITD']):
  if df_fleet[c].dtypes == 'O' and c not in ['date']:
        ax = fig.add_subplot(5, 4, i)
        qtd = df_fleet[c].value_counts()[:10]
        sns.boxplot(y ='total',x= c,data = df_fleet.loc[df_fleet[c].isin(qtd.index),:],ax = ax)#.set_xticklabels(df_fleet[col].unique(), rotation=45, ha='right')
        plt.xticks(rotation=45,ha='right')
        plt.xlabel('')
        plt.title(c)
        i +=  1

**BOXPLOT ANALYSIS**


*   It can be seen that the recognized views are unreadable for some reasons:
    - 'TOTAL' column are aproximated to 0
    - There are some outliers that is imparing the view
*   With the data in this situation, it is only possible to analyze the most striking outliers, such as:
    - Branch in unipetro-TP there is a very different amount in relation to the others
    - FLE4103 board there is a very outlier
    - VW and VOLVO brands have higher expenses compared to other brands

**We can apply data normalization with LOG to improve visualizations and adjust some outlier values**

#Appling LOG in some columns
df_fleet['total_log'] = df_fleet.total.apply(lambda x: 0 if np.isinf(np.log(x)) else np.log(x) )
df_fleet['quantity_log'] = df_fleet.quantity.apply(lambda x: 0 if np.isinf(np.log(x)) else  np.log(x))
df_fleet['capacity_log'] = df_fleet.capacity.apply(lambda x: 0 if np.isinf(np.log(x)) else np.log(x) )

##BOXPLOT TO VERIFY VARIATION AND OUTLIER OF NUMERICAL VARIABLES
i = 1
fig = plt.figure(figsize = (20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for j,c in enumerate(df_fleet.columns):
    if df_fleet[c].dtypes != 'O' and c not in ['date','year','document','tax_code']:
        ax = fig.add_subplot(5, 4, i)
        sns.boxplot(y = c,data = df_fleet,ax = ax)
        plt.title(c)
        i +=  1

i = 1
fig = plt.figure(figsize = (25,30))
fig.subplots_adjust(hspace=1.0, wspace=0.4)
for j,c in enumerate(df_fleet.columns):
  if df_fleet[c].dtypes == 'O' and c not in ['date']:
        ax = fig.add_subplot(5, 3, i)
        qtd = df_fleet[c].value_counts()[:10]
        sns.boxplot(y ='total_log',x= c,data = df_fleet.loc[df_fleet[c].isin(qtd.index),:],ax = ax)#.set_xticklabels(df_fleet[col].unique(), rotation=45, ha='right')
        plt.xticks(rotation=45,ha='right')
        plt.xlabel('')
        plt.title(c)
        i +=  1
plt.show()

#Distribution analysis with histogram


*   Model_year distribution
*   Quantity distribution
*   Total distribution

df_fleet.drop(columns=['document','tax_code']).hist(figsize=(20,10))
plt.show()

**ANÁLISE DOS HISTOGRAMAS**


*   outlier in quantity, total, capacity
*   higher frequency of records between the month 1 and 2, 11 and 12
*   vehicles between 2010 and 2020 are the majority

## DROPPING THE OUTLIER
df_fleet_clean = df_fleet.query("quantity < 350 and quantity > 1 and total < 3500 and total > 0")
print(df_fleet_clean.shape)

## CHECKING THE DISTRIBUTION OF VARIABLES
df_fleet_clean.drop(columns=['document','tax_code']).hist(figsize=(20,10))
plt.show()

##**PAIR PLOT**

Correcting the outliers based on previus analysis

## function to indentify the outliers based in the lower and upper limits

def get_limits_quantile(data):
    q1 = data.quantile(.25)
    q3 = data.quantile(.75)

    sup = q3 + 1.5 * (q3-q1)
    inf = q1 - 1.5 * (q3-q1)

    return inf, sup
print('OK')

#sns.pairplot(df_fleet[numerical_columns] )
pd.plotting.scatter_matrix(df_fleet_clean.drop(columns=['document','tax_code','month','year']), figsize=(20,15))
plt.show()

#Heatmap correlation analysis

plt.figure(figsize=(10,8))
cor_pearson = df_fleet_clean.drop(columns=['document','tax_code']).corr()
mask = np.triu(np.ones_like(cor_pearson, dtype=np.bool))
mask = mask[1:, :-1]
corr = cor_pearson.iloc[1:,:-1].copy()

sns.heatmap(corr,annot=True, linewidths=.3,vmin=-1, vmax=1, mask=mask)

**PEARSON CORRELATION ANALYSIS**


*   The Y(total) variable has quantity and capacity correlation
*   The Y(total_log) LOG variable has quantity, capacity and capacity_log correlaton
*   Indicative that we can develop linear regression models with normal and normalized variables with LOG

# CORRELAÇÃO DE SPEARMAN
plt.figure(figsize=(10,8))
cor_sperman = df_fleet_clean.drop(columns=['document','tax_code']).corr('spearman')
mask = np.triu(np.ones_like(cor_sperman, dtype=np.bool))
mask = mask[1:, :-1]
corr = cor_sperman.iloc[1:,:-1].copy()

sns.heatmap(corr,annot=True, linewidths=.3,vmin=-1, vmax=1, mask=mask)

**ANALYSIS OF SPEARMAN CORRELATIONS**
There was no significant change in any correlation of total and total_log

df_fleet_clean.to_csv('df_fleet_final.csv',index = False)

df_fleet.shape, df_fleet_clean.shape

#**CLUSTERS**
#instal yellowbricks for clustering
!pip install yellowbrick=='1.2.1'

from yellowbrick.cluster import KElbowVisualizer ## ELBOW Method
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

## defining color palette for graphs
sns.set_palette('tab10')

##Data collection and preparation for cluster
## COLLECTING DATA
df = pd.read_csv("df_fleet_final.csv")
#df_clean = df.drop(['document', 'tax_code', 'total_log', 'quantity_log', 'capacity_log', 'area_descption', 'date','license'])
## SELECT THE IMPORTANT COLUMNS FOR CLUSTER MODEL

df_clean = df[['branch_name', 'tax_description', 'quantity', 'total',
       'brand_name', 'vehicle_type', 'truck_type', 'department',
       'model_year', 'capacity', 'aplication', 'afiliate_name', 'month']]

df_clean.shape
df_clean.head(1)

df_clean.aplication.unique()

##**Convert categorical variables (columns) to one-hot-encoding (get_dummies)**



*   Algorithm works only with numeric variables, so it will be necessary to convert text variables to numeric variables
*   The strategy adopted will be the application of one-hot-encoding, transforming the variables into columns and indicating with 1 (when there is the respective value) and 0 (when there is no value)

one = OneHotEncoder()
data_cluster = pd.DataFrame(one.fit_transform(df_clean[['branch_name', 'tax_description',
                                                         'brand_name', 'vehicle_type', 'truck_type', 'department',
                                                         'aplication', 'afiliate_name']]).toarray(),
                             columns=one.get_feature_names())
data_cluster = pd.concat([df_clean[['quantity', 'total','model_year', 'capacity', 'month']],data_cluster],axis=1)

data_cluster.shape
data_cluster.head(3)

##**Data Normatization**

*   Normalizing the data to be on the same scale, this helps the algorithm identify similar data to assemble the clusters.

##**CLUSTER KMEANS**
###Better cluster number indentify



*   This analysis helps to identify the best cluster number for the kmeans algorithm
*   The yellowbricks package already has functions that perform this analysis
*   The method used will be **ELBOW**

elbow = KElbowVisualizer(KMeans(random_state=42), k=(2,12))## EXECUTING AND VIZUALIZING THE ELBOW RESULT
elbow.fit(data_cluster_nor)
elbow.show()

## BASED ON THE PREVIOUS RESULT, TRAIN THE FINAL CLUSTER ALGORITHM USING KMEANS
## KMEANS WITH 5 CLUSTER
kmeans = KMeans(n_clusters= 5,  random_state = 42)

## TRAINING AND DEFINITION OF CLUSTERS WITH KMEANS
clusters_m = kmeans.fit_predict(data_cluster_nor)

## PCA METHOD USED TO DECREASE DATABASE DIMENSIONS TO 2 DIMENSION
#THIS METHOD IS USED ONLY TO VIEW PREVIOUSLY TRAINED CLUSTERS
pca = PCA(n_components=2)
df_pca = pca.fit_transform(data_cluster_nor)
df_pca = pd.DataFrame(df_pca)
df_pca['cluster'] = clusters_m
print("CLUSTERS RECORDS: \n", df_pca['cluster'].value_counts())
plt.figure(figsize=(10,5))
sns.scatterplot(x=df_pca[0], y=df_pca[1], hue=df_pca['cluster'],palette='tab10', alpha=.5)

**PCA CLUSTER ANALYSIS**
 - It can be seen that the clusters were well divided
 - Clusters 0, 3 and 4 called quite separate, mean that the data is very similar for each cluster
 - Clusters 1 and 2 have already been mixed, indicating that the algorithm was not able to separate these two clusters as well
 - Overall the algorithm obtained a good result.
##CLUSTER ANALYSIS
 - A descriptive analysis will be applied in the cluster to identify any profile, pattern and the like

df['cluster'] = clusters_m

plt.figure(figsize=(10,5))
sns.countplot(x = 'cluster', data = df ,color=sns.color_palette('tab10')[0])
plt.title("Number of records in each cluster")
plt.show()

 **Clusters Records **
  - Cluster 3 has the lowest records
  - Cluster 1 has more records numbers
  - Cluster are well distributed

df.rename(columns={'quantidade':'quantity', 'ano_modelo':'model_year','capacidade':'capacity'},inplace=True)

i = 1
fig = plt.figure(figsize = (20,10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for c in ['quantity','total','model_year','capacity']: # df.columns:
      ax = fig.add_subplot(3, 2, i)
      sns.boxplot(y = c,data = df,x='cluster',ax = ax,showfliers = True)#,palette=paleta)
      plt.title(c)
      i +=  1

**BOXPLOT ANALYSIS**

It can be seen in QUANTITY that the clusters have a slight difference in the mediated (center lines of the box) between them

Cluster 0, 4 and 3 with a similar behavior for QUANTITY.

As for TOTAL, clusters 0, 1 and 2 have a very similar behavior

It is possible to observe that vehicles with capacity value cluster 0 and 2 have some similarity.

For model year has more variaty of records with the median allways above 2010.

fig = plt.figure(figsize = (25,10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
i=1
for c in ['quantity','total']: # df.columns:
  ax = fig.add_subplot(2, 2, i)
  sns.histplot(data = df,x=c,hue='cluster', alpha=.3, palette='tab10',ax=ax)
  plt.title(f"Histogram {c} by cluster")
  i +=  1
plt.show()

df.rename(columns={'documento':'document','placa':'license','data':'date','ano_modelo':'model_year','centro_custo_codigo':'tax_code','ano':'year','quantidade_log':'quantity_log','capacidade_log':'capacity_log'},inplace=True)

# ***************OPTIONAL**************
## FIRST OPTION - I can choose one of then

mean_grup = df.groupby('cluster').mean().reset_index().drop(columns=['document','tax_code','year','total_log','quantity_log','capacity_log'])

mean_grup_2 = mean_grup.melt('cluster',var_name='type',value_name='media')
plt.figure(figsize = (30,10))
g = sns.barplot(y = 'media',x='type', data=mean_grup_2,hue='cluster')
g.set_yscale("log")
plt.ylabel('')
plt.ylabel('')
g.set(yticklabels=[])
g.tick_params(axis='x', labelsize=14)
plt.title('Cluster Comparative')

for p in g.patches:
    g.annotate(format(p.get_height(), '.1f'),
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha = 'center', va = 'center',
               xytext = (0, 9), fontsize=13,
               textcoords = 'offset points')
plt.show()

## MORE DETAILED ANALYSIS OF EACH CLUSTER, WITH THE EXACT VALUES
# IT IS NOT NECESSARY TO INSERT ALL COLUMNS, IT CAN BE CHANGED TO KEEP ONLY THE MOST IMPORTANT COLUMNS AND FOCUSING ON THE BUSINESS
tabel_cluster = pd.DataFrame()
list_cluster = []

for g in df.cluster.value_counts().index: #unique():
  dicionario = dict()
  dicionario['cluster'] = g

  for c in df.columns:
    temp = df.query(f"cluster == {g}")
    if temp[c].dtype == 'O' and c not in ['license','date','tax_description']:
      dicionario[c] = temp[c].value_counts().index[:3].tolist()

  list_cluster.append(dicionario)

pd.DataFrame( list_cluster ).sort_values('cluster')

# Option two
mean_grup = df.groupby('cluster').mean().reset_index().drop(columns=['document','tax_code','year','total_log','quantity_log','capacity_log'])

mean_grup_2 = mean_grup.melt('cluster',var_name='type',value_name='media')
plt.figure(figsize = (30,10))
g = sns.barplot(y = 'media',x='type', data=mean_grup_2,hue='cluster')
g.set_yscale("log")
plt.ylabel('')
plt.ylabel('')
g.set(yticklabels=[])
g.tick_params(axis='x', labelsize=14)
plt.title('Cluster Comparative')

for p in g.patches:
    g.annotate(format(p.get_height(), '.1f'),
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha = 'center', va = 'center',
               xytext = (0, 9), fontsize=13,
               textcoords = 'offset points')
plt.show()

 ## MORE DETAILED ANALYSIS OF EACH CLUSTER, WITH THE EXACT VALUES
# IT IS NOT NECESSARY TO INSERT ALL COLUMNS, IT CAN BE CHANGED TO KEEP ONLY THE MOST IMPORTANT COLUMNS FOR AND FOCUSING ON THE BUSINESS
for g in df.cluster.value_counts().index: #unique():
  print("="*10,f"Cluster: {g}", "="*10,'\n')
  for c in df.columns: #['branch_name','quantity']:
    temp = df.query(f"cluster == {g}") #.groupby(['cluster']).count()
    if temp[c].dtype == 'O' and c not in ['license','date']:
      print(f"==> {c}: {temp[c].value_counts().index[:3].tolist() } ")

    elif c in ['quantity', 'total', 'model_year','capacity', 'month']:
      print(f"==> {c}: {np.round(temp[c].mean()) }(media) ")

  print()

  ## FUNCTION TO ENCAPSULATE TRAINING AND INITIAL VALIDATION OF THE MODEL
def make_train_model(models, X_train,X_test,y_train,y_test):
  for model in models:
    print("="*5, model.__class__.__name__,"="*5)
    model_fit = model.fit(X_train,y_train)
    pred_train = model_fit.predict(X_train)

    ## VALIDATION TRAIN
    print("==> TRAIN VALIDATION")
    print('MSE : {}'.format(mean_squared_error(y_true = y_train, y_pred = pred_train)))
    print('R2 : {:.2f}%'.format(r2_score(y_true = y_train, y_pred = pred_train)*100 ))
    print()

    pred_test = model_fit.predict(X_test)
    print("==> TEST VALIDATION")
    print('MSE : {}'.format(mean_squared_error(y_true = y_test, y_pred = pred_test)))
    print('R2 : {:.2f}%'.format(r2_score(y_true = y_test, y_pred = pred_test)*100 ))
    print()

    from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import  r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import seaborn as sns

##Data preparation
 - transform categorical columns
 - eliminate columns with high Nun values

df = pd.read_csv("df_fleet_final.csv")
df.columns

pd.plotting.scatter_matrix(df.drop(columns=['document','tax_code','month','year']).query("quantity > 50 and total < 300 "), figsize=(20,15))
plt.show()

Regression Models

 - the objective of the models is to predict the amount spent (total) from other variables.
 - The following algorithms were tested: Decision Tree Regressor, Linear Regresseion and Ridge

##Simple Linear Regression
 - Variables:
   - total(y) and capacity(x)
   - total(y) and year_model(x)
  
## MODEL EXECUTION
## SPLITTING DATA IN TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(df[['capacity']], df['total'], test_size= 0.2)

## CREATING LIST WITH TEMPLATES TO BE USED
model_list = [DecisionTreeRegressor(), LinearRegression(), Ridge()]

## RUNNING THE FUNCTION TO TRAIN AND VALIDATE THE MODEL
make_train_model(model_list,X_train,X_test,y_train,y_test)

#All cells below perform the same procedure as the cell above.
X_train, X_test, y_train, y_test = train_test_split(df[['model_year']], df['total'], test_size= 0.2)
model_list = [DecisionTreeRegressor(), LinearRegression(), Ridge()]
make_train_model(model_list,X_train,X_test,y_train,y_test)

##Simple linear regression with LOG columns

 - tota_log(y) and capacity_log(x)

X_train, X_test, y_train, y_test = train_test_split(df[['capacity_log']], df['total_log'], test_size= 0.2)
model_list = [DecisionTreeRegressor(), LinearRegression(), Ridge()]
make_train_model(model_list,X_train,X_test,y_train,y_test)

cluster_data = pd.DataFrame(MinMaxScaler().fit_transform(df[['quantity','total']]))
X_train, X_test, y_train, y_test = train_test_split(cluster_data.iloc[:,[0]], cluster_data.iloc[:,1], test_size= 0.2)
model_list = [DecisionTreeRegressor(), LinearRegression(), Ridge()]
make_train_model(model_list,X_train,X_test,y_train,y_test)

##Multiple Linear Regression

 - total(y) and capacity, model_year,branch_name, brand_name, vehicle type, truck_type, aplication

## CONVERTING CATEGORY VALUES TO COLUMN  (dummies)
one = OneHotEncoder()
data_one = pd.DataFrame(one.fit_transform(df[['brand_name', 'vehicle_type', 'truck_type', 'department']]).toarray(),
                             columns=one.get_feature_names())
data_one = pd.concat([df[['total', 'capacity','quantity']],data_one],axis=1)
data_one.head()

x = data_one.drop(columns=['total'])
y = data_one.total
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
model_list = [DecisionTreeRegressor(), LinearRegression(), Ridge()]
make_train_model(model_list,X_train,X_test,y_train,y_test)

