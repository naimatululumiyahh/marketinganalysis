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


#check insights of the data (Customer Age)
f_train = df_train[df_train['Year_Birth'] > 1900].reset_index(drop=True)
df_validation = df_validation[df_validation['Year_Birth'] > 1900].reset_index(drop=True)
df_test = df_test[df_test['Year_Birth'] > 1900].reset_index(drop=True)

df_train['Customer_Age']  = df_train['Dt_Customer'].dt.year - df_train['Year_Birth']
df_validation['Customer_Age']  = df_validation['Dt_Customer'].dt.year - df_validation['Year_Birth']
df_test['Customer_Age']  = df_test['Dt_Customer'].dt.year - df_test['Year_Birth']
#Conclution: average customer age in 40 years old

#check insights of the data (Marital Status)
df_train['Marital_Status'] = df_train['Marital_Status'].apply(lambda x: 'Single' if str(x) in ['YOLO', 'Alone' , 'Absurd'] else str(x))
df_validation['Marital_Status'] = df_validation['Marital_Status'].apply(lambda x: 'Single' if str(x) in ['YOLO', 'Alone' , 'Absurd'] else str(x))
df_test['Marital_Status'] = df_test['Marital_Status'].apply(lambda x: 'Single' if str(x) in ['YOLO', 'Alone' , 'Absurd'] else str(x))

#check insight of data (Kidhome and Teenhome)
df_train['Num_Dependants'] = df_train['Kidhome'] + df_train['Teenhome']
df_validation['Num_Dependants'] = df_validation['Kidhome'] + df_validation['Teenhome']
df_test['Num_Dependants'] = df_test['Kidhome'] + df_test['Teenhome']

#check insight of data to know how long the customer has been joined the company
#we will create new features for the month and year of the customer joined the company
df_train['Dt_Customer_Month'] = df_train['Dt_Customer'].dt.month
df_train['Dt_Customer_Year'] = df_train['Dt_Customer'].dt.year
df_validation['Dt_Customer_Month'] = df_validation['Dt_Customer'].dt.month
df_validation['Dt_Customer_Year'] = df_validation['Dt_Customer'].dt.year
df_test['Dt_Customer_Month'] = df_test['Dt_Customer'].dt.month
df_test['Dt_Customer_Year'] = df_test['Dt_Customer'].dt.year

#check insight of data to know average amount spent by customer
amt_spent_features = [ c for c in df.columns if 'Mnt' in str(c)]
df_train['TotalAmount_Spent'] = df_train[amt_spent_features].sum(axis=1)
df_validation['TotalAmount_Spent'] = df_validation[amt_spent_features].sum(axis=1)
df_test['TotalAmount_Spent'] = df_test[amt_spent_features].sum(axis=1)

#check insight of data purchase by customer
purchase_feature = [c for c in df.columns if 'Purchase' in str(c)]
df_train['Total_Purchases'] = df_train[purchase_feature].sum(axis=1)
df_validation['Total_Purchases'] = df_validation[purchase_feature].sum(axis=1)
df_test['Total_Purchases'] = df_test[purchase_feature].sum(axis=1)

#Feature Selection
#we will select the features that we will use for the model
cmp_feat = [c for c in df.columns if 'AcceptedCmp' in str(c)]
mnt_feat = [c for c in df.columns if 'Mnt' in str(c)]
num_feat = [c for c in df.columns if 'Num' in str(c)]
numeric_feat = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'Complain', 'Customer_Age',
                'Num_Dependants', 'Dt_Customer_Month', 'Dt_Customer_Year', 
                'TotalAmount_Spent', 'Total_Purchases']
all_numeric_feat = cmp_feat + mnt_feat + num_feat + numeric_feat


#Preprocessing Data (CATEGORICAL PROCESSING)
#we will convert the categorical features to numerical features using one-hot encoding

#we will select the categorical features that we will use for the model
categoric_feat = ['Education', 'Marital_Status', 'Country']
all_feat = categoric_feat + all_numeric_feat
df_train_final = df_train[all_feat]
df_validation_final = df_validation[all_feat]
df_test_final = df_test[all_feat]

#we have 3 categorical features: Education, Marital_Status and Country
#we will use one-hot encoding to convert the categorical features to numerical features

#Education
df_train.Education.unique() #check unique values
education = {'Basic':1 , 'Graduation':2, '2n Cycle':3, 'Master':4 , 'PhD':5}

#we will map the Education feature to numerical values
df_train_final['Education'] = df_train_final['Education'].map(education)
df_validation_final['Education'] = df_validation_final['Education'].map(education)
df_test_final['Education'] = df_test_final['Education'].map(education)

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

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
# X = independent variable
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

#PERFORMANCE STABILITY CHECK
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


