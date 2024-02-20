#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import missingno as msno
import scipy.stats as stats
#creating interactive dashboards
import plotly.express as px
#estimating and testing statistical models
import statsmodels.api as sm
#for extra large datasets 
import dask.dataframe as dd
#chisquare
from scipy.stats import chi2_contingency


# In[2]:


df= pd.read_excel('Amazon Sale Report.xlsx')


# In[3]:


df.sample(5)


# In[4]:


df.isna().sum()


# In[5]:


df.shape


# In[6]:


#make the entries caps for uniformity
df['ship_city1'] = df['ship-city'].str.upper()
df['ship_state1'] = df['ship-state'].str.upper()


# In[7]:


#check for duplicates using the unique id
#check for duplicates
duplicates = df.duplicated(subset=['Order ID']).any()

# Display the result
print("Duplicates:", duplicates)


# In[8]:


#keep last instance
df1 = df.drop_duplicates(subset=['Order ID'], keep = 'last')
#check for duplicates
duplicates = df1.duplicated(subset=['Order ID']).any()

# Display the result
print("Duplicates:", duplicates)


# In[9]:


df1.shape


# In[10]:


#impute missing values
#Country and currency should be IN and INR respectively
#It is speculated that courier state is directly dependent on shipstate while amount is directly dependent on category and ship city
#we'll use chi square to check for independence


# In[11]:


# Fill missing values in the 'currency' column with 'INR' & in the 'country' column with 'IN'
df1['currency'].fillna('INR', inplace=True)
df1['ship-country'].fillna('IN', inplace=True)


# In[44]:


# Define the variable pairs for chi-square tests
variable_pairs = [('ship_state1', 'ship_city1'), ('ship_city1', 'Courier Status'), ('Amount', 'Category'), ('Amount', 'ship_city1'), ('Courier Status','Status')]


# In[45]:


# Perform chi-square tests
for var1, var2 in variable_pairs:
    contingency_table = pd.crosstab(df1[var1], df1[var2])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for Independence between '{var1}' and '{var2}':")
    print(f"P-value: {p_value}")
    print("")


# In[14]:


#a p-value >0.05 suggests independence between the two variables


# In[15]:


# Create a copy of the DataFrame to store imputed values
df_imputed = df1.copy()


# In[30]:


# Step 1: Impute missing values in 'ship-state' with the most repeated state(do the same for ship-state)
most_repeated_state = df_imputed['ship_state1'].mode()[0]
df_imputed['ship_state1'].fillna(most_repeated_state, inplace=True)


# In[31]:


# Step 2: Impute missing values in 'ship-city' with the most repeated city in each chosen state
most_repeated_city_per_state = df_imputed.groupby('ship_state1')['ship_city1'].transform(lambda x: x.mode()[0])
df_imputed['ship_city1'].fillna(most_repeated_city_per_state, inplace=True)


# In[32]:


missing_postal_codes = df_imputed[df_imputed['ship-postal-code'].isnull()]

# Selecting the columns for postal code, city, and state
missing_postal_codes_info = missing_postal_codes[['ship-postal-code', 'ship_city1', 'ship_state1']]

# Display the DataFrame with missing postal codes
print(missing_postal_codes_info)


# In[21]:


mumbai_zipcodes = df_imputed[df_imputed['ship-city'] == 'Mumbai']['ship-postal-code'].drop_duplicates()

# Print the unique postal codes where city is Mumbai
print(mumbai_zipcodes)


# In[22]:


#Calculate the median of the zip codes for Mumbai
mumbai_median_zipcode = df_imputed[df_imputed['ship-city'] == 'Mumbai']['ship-postal-code'].median()

#Replace the missing values in the "zipcode" column where the city is "Mumbai" with the calculated median
df_imputed.loc[(df['ship-city'] == 'Mumbai') & (df_imputed['ship-postal-code'].isnull()), 'ship-postal-code'] = mumbai_median_zipcode


# In[25]:


#Calculate the mode of the entire "zipcode" column
zipcode_mode = df_imputed['ship-postal-code'].mode()[0]

#Replace the remaining missing values in the "zipcode" column with the calculated mode
df_imputed['ship-postal-code'].fillna(zipcode_mode, inplace=True)


# In[40]:


#Impute missing values in 'Amount' with the mean amount according to category and ship city
mean_amount_per_category_and_city = df_imputed.groupby(['Category', 'ship_city1'])['Amount'].transform('mean')
df_imputed['Amount'].fillna(mean_amount_per_category_and_city, inplace=True)


# In[42]:


#Impute remaining missing values in 'Amount' with the mean amount according to category
mean_amount_per_category_and_city = df_imputed.groupby(['Category'])['Amount'].transform('mean')
df_imputed['Amount'].fillna(mean_amount_per_category_and_city, inplace=True)


# In[47]:


#impute courier status using the column status
most_Cs_by_status = df_imputed.groupby('Status')['Courier Status'].transform(lambda x: x.mode()[0])
df_imputed['Courier Status'].fillna(most_Cs_by_status, inplace=True)


# In[49]:


#fill missing values in promotion-ids and fulfilled-by
df_imputed['promotion-ids'].fillna('NA', inplace=True)
df_imputed['fulfilled-by'].fillna('Easy Ship', inplace=True)


# In[51]:


#delete unnecessary columns
df_imputed.drop(columns=['Unnamed: 22'], inplace=True)


# In[52]:


df_imputed.isna().sum()


# In[54]:


#download dataset ---visualization in PowerBI
df_imputed.to_csv('Amazonnew.csv', index = False)


# In[ ]:




