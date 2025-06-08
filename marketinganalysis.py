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
#to evaluate the model performance, we need to split the data into training, validation and testing sets
#we need to import the train_test_split function from sklearn.model_selection
from sklearn.model_selection import train_test_split

#we will split data into 60% training, 20% validation and 20% testing
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_validation = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train.shape


