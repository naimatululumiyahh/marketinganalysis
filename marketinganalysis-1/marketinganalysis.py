import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_palette('Spectral')

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('marketing_data.csv')

#Data Splitting
#to evaluate the model performance, we need to split the data into training, ationation and testing sets
#we need to import the train_test_split function from sklearn.model_selection
from sklearn.model_selection import train_test_split

#we will split data into 60% training, 20% validation and 20% testing
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_validation = train_test_split(df_full_train, test_size=0.25, random_state=1)
print(df_train.shape, df_validation.shape, df_test.shape)

#now we have three datasets: df_train, df_validation and df_test
#we will use df_train to train the model, df_validation to tune the model and df_test to evaluate the model performance

#Preprocessing Data (NUMERICAL PROCESSING)

#1.feature datatypes
#we will check the datatypes of the features in each dataset
print(df_train.info(), '\n', df_validation.info(), '\n', df_test.info())    

#conclusion: we have 3 error from data ()
#1.1 there is whitespace in column 'Income' so we need to remove it
#1.2 in column 'Income', datatype is object so we need to convert it to numeric (float64)
#1.3 in column 'Dt_Customer', datatype is object so we need convert to datetime


#1.1 remove whitespace in column 'Income'
df_train.columns = df_train.columns.str.strip()
df_validation.columns = df_validation.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

#1.2 Convert datatype 'Income' to float64
#before we convert the datatype, we need to remove symbol '$' and ','
df_train['Income'] = df_train['Income'].str.replace('$','')
df_train['Income'] = df_train['Income'].str.replace(',','')
df_validation['Income'] = df_validation['Income'].str.replace('$','') 
df_validation['Income'] = df_validation['Income'].str.replace(',','')
df_test['Income'] = df_test['Income'].str.replace('$','')
df_test['Income'] = df_test['Income'].str.replace(',','')
#after removing the symbol, we can convert the datatype to float64
df_train['Income'] = df_train['Income'].astype('float64')
df_validation['Income'] = df_validation['Income'].astype('float64')
df_test['Income'] = df_test['Income'].astype('float64')

#1.3 convert datatype 'Dt_Customer' to datetime
#we must convert wit pd.to_datetim() function cz we must to convert the object to datetime
df_train['Dt_Customer'] = pd.to_datetime(df_train['Dt_Customer'])
df_validation['Dt_Customer'] = pd.to_datetime(df_validation['Dt_Customer'])
df_test['Dt_Customer'] = pd.to_datetime(df_test['Dt_Customer'])

#1.3 remove whitespace in column 'Income'
df_train.columns = df_train.columns.str.strip()
df_validation.columns = df_validation.columns.str.strip()
df_test.columns = df_test.columns.str.strip()


#2. Duplicate, Missing Values and Outliers
#2.1 check for duplicate rows
print(df_train.duplicated().sum())
print(df_validation.duplicated().sum())
print(df_test.duplicated().sum())

#we dont have any duplicate rows in any dataset

#2.2 check for missing values
print(df_train.isna().sum().sort_values(ascending=False))
print(df_validation.isna().sum().sort_values(ascending=False))
print(df_test.isna().sum().sort_values(ascending=False))


#conclusion: we have missing values in column Income

#check distribution of Income (Data Training)
sns.displot(df_train['Income'], kde= False)
df_train['Income'].plot(kind='box', figsize=(3,4))
#we have distribution not normal, so we will use median to fill the missing values
df_train['Income'] = df_train['Income'].fillna(df_train['Income'].median())

#check distribution of Income (Data Validation)
sns.displot(df_validation['Income'], kde = False)
df_validation['Income'].plot(kind = 'box', figsize = (3,4))
#we have distribution not normal, so we will use median to fill the missing values
df_validation['Income'] = df_validation['Income'].fillna(df_validation['Income'].median())

#check distribution of Income (Data Testing)
sns.displot(df_test['Income'], kde = False)
df_test['Income'].plot(kind = 'box', figsize = (3,4))
#we have distribution not normal, so we will use median to fill the missing values
df_test['Income'] = df_test['Income'].fillna(df_test['Income'].median())

#2.3 check for outliers
df_train_to_plot = df_train.select_dtypes(include=['float','int'])
df_train_to_plot.drop(columns=['ID','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                               'AcceptedCmp4','AcceptedCmp5','Response','Complain'], 
                      inplace=True)
df_train_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(14,16))
plt.subplots_adjust(wspace=0.5)

df_validation_to_plot = df_validation.select_dtypes(include=['float','int'])
df_validation_to_plot.drop(columns=['ID','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                                    'AcceptedCmp4','AcceptedCmp5','Response','Complain'], 
                           inplace=True)
df_validation_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(14,16))
plt.subplots_adjust(wspace=0.5)

df_test_to_plot = df_test.select_dtypes(include=['float','int'])
df_test_to_plot.drop(columns=['ID','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                              'AcceptedCmp4','AcceptedCmp5','Response','Complain'], 
                     inplace=True)
df_test_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(14,16))
plt.subplots_adjust(wspace=0.5)

#we have outliers in the following columns:
#there are customer birth year before 1900, so we will remove those rows
#we will remove the rows with Year_Birth < 1900
df_train = df_train[df_train['Year_Birth']>1900].reset_index(drop=True)
df_validation = df_validation[df_validation['Year_Birth']>1900].reset_index(drop=True)
df_test = df_test[df_test['Year_Birth']>1900].reset_index(drop=True)

#DATA INSIGHTS
# before we get conclusion, we must to check percentage of customers who accepted the offer
# make it if just one column:
#ax = sns.countplot(x='Marital_Status', hue='Response', data=df)
#barPerc(df, 'Marital_Status', ax)
# make it if we grouping multiple columns:
#sns.histplot(df, x='Marital_Status', hue='Response', multiple='stack')

def barPerc(df, col, ax):
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{height/total:.2%}'
        x = p.get_x() + p.get_width() / 2
        y = height
        ax.annotate(percentage, (x, y), ha='center', va='bottom')
#check insights of the data (Response)

#check insights of the data (Customer Age)
df_train = df_train[df_train['Year_Birth'] > 1900].reset_index(drop=True)
df_train['Customer_Age']  = df_train['Dt_Customer'].dt.year - df_train['Year_Birth']
ax = sns.countplot(x='Year_Birth', hue='Response', data=df_train)
barPerc(df_train, 'Year_Birth', ax)
df_validation = df_validation[df_validation['Year_Birth'] > 1900].reset_index(drop=True)
df_validation['Customer_Age']  = df_validation['Dt_Customer'].dt.year - df_validation['Year_Birth']
ax = sns.countplot(x='Year_Birth', hue='Response', data=df_validation)
barPerc(df_validation, 'Year_Birth', ax)
df_test = df_test[df_test['Year_Birth'] > 1900].reset_index(drop=True)
df_test['Customer_Age']  = df_test['Dt_Customer'].dt.year - df_test['Year_Birth']
ax = sns.countplot(x='Year_Birth', hue='Response', data=df_test)
barPerc(df_test, 'Year_Birth', ax)
#Conclution: average customer age in 40 years old take the offer

#check insights of the data (Marital Status)
df_train['Marital_Status'] = df_train['Marital_Status'].apply(lambda x: 'Single' 
                                                              if str(x) in ['YOLO', 'Alone' , 'Absurd'] 
                                                              else str(x))
ax = sns.countplot(x='Marital_Status', hue='Response', data=df_train)
barPerc(df_train, 'Marital_Status', ax)
df_validation['Marital_Status'] = df_validation['Marital_Status'].apply(lambda x: 'Single' 
                                                                        if str(x) in ['YOLO', 'Alone' , 'Absurd'] 
                                                                        else str(x))
ax = sns.countplot(x='Marital_Status', hue='Response', data=df_validation)
barPerc(df_validation, 'Marital_Status', ax)
df_test['Marital_Status'] = df_test['Marital_Status'].apply(lambda x: 'Single' 
                                                            if str(x) in ['YOLO', 'Alone' , 'Absurd'] 
                                                            else str(x))
ax = sns.countplot(x='Marital_Status', hue='Response', data=df_test)
barPerc(df_test, 'Marital_Status', ax)
#conclusion: customer marital status 'Single' in every dataset is the highest percetages take the offer

#check insight of data (Kidhome and Teenhome)
df_train['Num_Dependants'] = df_train['Kidhome'] + df_train['Teenhome']
ax = sns.countplot(x='Num_Dependants', hue='Response', data=df_train)

barPerc(df_train, 'Num_Dependants', ax)
df_validation['Num_Dependants'] = df_validation['Kidhome'] + df_validation['Teenhome']
ax = sns.countplot(x='Num_Dependants', hue='Response', data=df_validation)
barPerc(df_validation, 'Num_Dependants', ax)
df_test['Num_Dependants'] = df_test['Kidhome'] + df_test['Teenhome']
ax = sns.countplot(x='Num_Dependants', hue='Response', data=df_test)
barPerc(df_test, 'Num_Dependants', ax)
#conclusion: average number of dependants is 0, they accepted the offer (percentage of 0 dependants is the highest)

#check insight of data to know how long the customer has been joined the company
#we will create new features for the month and year of the customer joined the company
df_train['Dt_Customer_Month'] = df_train['Dt_Customer'].dt.month
df_train['Dt_Customer_Year'] = df_train['Dt_Customer'].dt.year
ax = sns.countplot(x='Dt_Customer_Month', hue='Response', data=df_train)
barpercentage = df_train['Dt_Customer_Month']
barPerc(df_train, 'Dt_Customer_Month', ax)
df_validation['Dt_Customer_Month'] = df_validation['Dt_Customer'].dt.month
df_validation['Dt_Customer_Year'] = df_validation['Dt_Customer'].dt.year
df_test['Dt_Customer_Month'] = df_test['Dt_Customer'].dt.month
df_test['Dt_Customer_Year'] = df_test['Dt_Customer'].dt.year
#conclusion: average customer has been joined the company for 8-9 months accepted the offer (highest percentage of 8-9 months)

#check insight of data to know the recency of the customer
sns.histplot(data=df_train, x='Recency', hue='Response', multiple='stack', kde=True)
sns.histplot(data=df_validation, x='Recency', hue='Response', multiple='stack', kde=True)
sns.histplot(data=df_test, x='Recency', hue='Response', multiple='stack', kde=True)
#conclusion: average recency of cosrtumer recently bought the product, accepted the offer (highest percentage of 0-1 recency)

#check insight of data to know average amount spent by customer
#group by the amount spent by customer with:
amt_spent_features_train = [ c for c in df_train.columns if 'Mnt' in str(c)]
amt_spent_features_train.append('Response')
df_train['TotalAmount_Spent'] = df_train[amt_spent_features_train].sum(axis=1)
sns.histplot(df_train, hue='Response', x='TotalAmount_Spent', multiple='stack', bins=30)
amt_spent_features_validation = [ c for c in df_validation.columns if 'Mnt' in str(c)]
amt_spent_features_validation.append('Response')
df_validation['TotalAmount_Spent'] = df_validation[amt_spent_features_validation].sum(axis=1)
sns.histplot(df_validation, hue='Response', x='TotalAmount_Spent', multiple='stack', bins=30)
amt_spent_features_test = [ c for c in df_test.columns if 'Mnt' in str(c)]
amt_spent_features_test.append('Response')
df_test['TotalAmount_Spent'] = df_test[amt_spent_features_test].sum(axis=1)
sns.histplot(df_test, hue='Response', x='TotalAmount_Spent', multiple='stack', bins=30)
#conclusion: average amount spent by customer is around 1000-2000, they accepted the offer (highest percentage of 1000-2000)

#check insight of data from NumwebvisitMonth feature
df_train[['NumWebVisitsMonth', 'Response']].corr()[['Response']]
df_validation[['NumWebVisitsMonth', 'Response']].corr()[['Response']]
df_test[['NumWebVisitsMonth', 'Response']].corr()[['Response']]
#conclusion: dont have correlation with Response, so we will not use this feature in the model

#check insight of data from Previous Campaigns
prev_cmp = [c for c in df.columns if 'AcceptedCmp' in str(c)]
prev_cmp.append('Response')
df_train[prev_cmp].corr()[['Response']].sort_values(by='Response', ascending=False)
df_validation[prev_cmp].corr()[['Response']].sort_values(by='Response', ascending=False)
df_test[prev_cmp].corr()[['Response']].sort_values(by='Response', ascending=False)

#check insight of data Complain
df_train[['Complain', 'Response']].corr()[['Response']]
df_validation[['Complain', 'Response']].corr()[['Response']]
df_test[['Complain', 'Response']].corr()[['Response']]
#conclusion: dont have correlation with Response, so we will not use this feature in the model


#check insight of data country
ax = sns.countplot(x='Country', hue='Response', data=df)
barPerc(df_train, 'Country', ax)
barPerc(df_validation, 'Country', ax)
barPerc(df_test, 'Country', ax)
#Mexico and Spain are the countries with the highest percentage of customers who accepted the offer

#check insight of data purchase by customer
purchase_feature_train = [c for c in df_train.columns if 'Purchase' in str(c)]
df_train['Total_Purchases'] = df_train[purchase_feature_train].sum(axis=1)
sns.histplot(df_train, hue='Response', x='Total_Purchases', multiple='stack', bins=30)
purchase_feature_validation = [c for c in df_validation.columns if 'Purchase' in str(c)]
df_validation['Total_Purchases'] = df_validation[purchase_feature_validation].sum(axis=1)
sns.histplot(df_validation, hue='Response', x='Total_Purchases', multiple='stack', bins=30)
purchase_feature_test = [c for c in df_test.columns if 'Purchase' in str(c)]
df_test['Total_Purchases'] = df_test[purchase_feature_test].sum(axis=1)
sns.histplot(df_test, hue='Response', x='Total_Purchases', multiple='stack', bins=30)
#customer with Total_Purchases between 5-25 is the highest percentage accepted the offer, highest peak is around 7-8 purchases


#Feature Selection
#we will select the features that we will use for the model
#to check the features that we have in each dataset
cmp_feat = [c for c in df.columns if 'AcceptedCmp' in str(c)]
mnt_feat = [c for c in df.columns if 'Mnt' in str(c)]
num_feat = [c for c in df.columns if 'Num' in str(c)]

numeric_feat = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'Complain', 'Customer_Age',
                'Num_Dependants', 'Dt_Customer_Month', 'Dt_Customer_Year', 
                'TotalAmount_Spent', 'Total_Purchases']
all_numeric_feat = cmp_feat + mnt_feat + num_feat + numeric_feat

print(df_train[all_numeric_feat].head(2))
print(df_validation[all_numeric_feat].head(2))
print(df_test[all_numeric_feat].head(2))
#to check the features that we have in each dataset, we will use the numeric features that we have selected


#Preprocessing Data (CATEGORICAL PROCESSING)
#we will convert the categorical features to numerical features using one-hot encoding

#we will select the categorical features that we will use for the model
categoric_feat = ['Education', 'Marital_Status', 'Country']
all_feat = categoric_feat + all_numeric_feat 

df_train_final = df_train[all_feat]
df_validation_final = df_validation[all_feat]
df_test_final = df_test[all_feat]
#we have 3 categorical features: Education, Marital_Status (nominal) and Country (nominal)
#we will use one-hot encoding to convert the categorical features to numerical features

#Education (ordinal feature)
df_train.Education.unique() #check unique values
education = {'Basic':1 , 'Graduation':2, '2n Cycle':3, 'Master':4 , 'PhD':5}

#we will map the Education feature to numerical values
df_train_final['Education'] = df_train_final['Education'].map(education)
df_validation_final['Education'] = df_validation_final['Education'].map(education)
df_test_final['Education'] = df_test_final['Education'].map(education)

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

#we must to change dataframe to dictionary format for DictVectorizer
df_train_final_dicts = df_train_final.to_dict(orient='records')
df_validation_final_dicts = df_validation_final.to_dict(orient='records')
df_test_final_dicts = df_test_final.to_dict(orient='records')

df_train_final_dicts = dv.fit_transform(df_train_final_dicts)
df_validation_final_dicts = dv.transform(df_validation_final_dicts)
df_test_final_dicts = dv.transform(df_test_final_dicts)

df_train_final = pd.DataFrame(df_train_final_dicts, columns = dv.get_feature_names_out())
df_validation_final = pd.DataFrame(df_validation_final_dicts, columns = dv.get_feature_names_out())
df_test_final = pd.DataFrame(df_test_final_dicts, columns = dv.get_feature_names_out())


#MODELLING
# X = independent variable (all variables)
# y = dependent variable (target variable) --> response 

X_train_final = df_train_final
X_validation_final = df_validation_final
X_test_final = df_test_final


y_train_final = df_train.Response
y_validation_final = df_validation.Response
y_test_final = df_test.Response


#BASE MODEL

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix   

model = LogisticRegression(random_state=42)
model.fit(X_train_final, y_train_final)

y_validation_pred = model.predict_proba(X_validation_final)[:,1]
print('LogisticRegression ROCAUC Result:' , roc_auc_score(y_validation_final, y_validation_pred), 3)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_final, y_train_final)

pred = model.predict_proba(X_validation_final)[:,1]
print('RandomForest ROCAUC Result:' , roc_auc_score(y_validation_final, y_validation_pred), 3)

#cause random forest have better ROCAUC result, we will use random forest as the base model

#PERFORMANCE STABILITY CHECK
#to check the performance stability, to use the model in the another dataset
X_full_train_final = pd.concat([X_train_final, X_validation_final])
y_full_train_final = pd.concat([y_train_final, y_validation_final])

model = RandomForestClassifier(random_state=42)
model.fit(X_full_train_final, y_full_train_final)

y_test_pred = model.predict_proba(X_test_final)[:,1]

print('RandomForest ROCAUC Result:' , roc_auc_score(y_test_final, y_test_pred), 3)


#INTERPRETATION
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test_final, y_test_final, n_repeats=10, random_state=42)
importance = result.importances_mean

# Plot
import matplotlib.pyplot as plt
import numpy as np

indices = np.argsort(importance)[-10:]  # Top 10
plt.figure(figsize=(12, 8))
plt.title("Top 10 Feature Importances (Permutation)")
plt.barh(range(10), importance[indices], align='center')
plt.yticks(range(10), [X_test_final.columns[i] for i in indices])
plt.xlabel("Mean Importance")
plt.tight_layout()
plt.show()