#!/usr/bin/env python
# coding: utf-8

# In[1]:


#cleaning analysis & vizualization
import numpy as np
import pandas as pd
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


# In[2]:


df = pd.read_csv("netflix1.csv")


# In[3]:


df.sample(10)


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


#check for duplicates
duplicates = df.duplicated(subset=['title']).any()

# Display the result
print("Duplicates:", duplicates)


# In[7]:


column_name = 'title'
duplicates = df[df.duplicated(subset=[column_name], keep=False)]

# Display the duplicate rows
print("Duplicate Rows:")
print(duplicates)


# In[20]:


#new dataframe
#keep first instance of duplicated rows
df1 = df.drop_duplicates(subset=['title'], keep = 'first')
#check for duplicates
duplicates = df1.duplicated(subset=['title']).any()

# Display the result
print("Duplicates:", duplicates)


# In[21]:


df1.sample(10)


# In[22]:


df1.shape


# In[23]:


#unique value analysis - how many different values e.g(there's 2 entries for gender[1 & 0])
dict = {}
for i in list(df1.columns):
    dict[i] = df1[i].value_counts().shape[0]
pd.DataFrame(dict, index = ["unique count"]).transpose()


# In[24]:


# Create a frequency distribution with release_year on the x-axis, differentiating by type
# Create bins for release years in increments of 5
bins = list(range(df1['release_year'].min(), df1['release_year'].max() + 6, 5))
labels = [f'{start}-{start+4}' for start in bins[:-1]]

# Add a new column 'release_year_bin' with the bin labels
df1['release_year_bin'] = pd.cut(df1['release_year'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(12, 6))
sns.countplot(x='release_year_bin', hue='type', data=df1, palette='Set1')
plt.title('Frequency Distribution of Movies and TV Shows Released Each 5-Year Bin')
plt.xlabel('Release Year Bin')
plt.ylabel('Count')
plt.legend(title='Type', loc='upper right')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()


# In[27]:


df1['date_added'] = pd.to_datetime(df1['date_added'], errors='coerce')  # Convert to datetime format
df1['month_added'] = df1['date_added'].dt.month_name()  # Extract month name


# In[28]:


df1.isna().sum()


# In[38]:


df1.sample(10)


# In[25]:


#movies vs tv shows
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x=df1["type"], palette="magma")
plt.show()


# In[26]:


##Chi Square Test of independence for the following pairs
# type and director
#type and month_added
#type and country
#rating and country
#H0:type and director are two independent variables
#HA:there is significant association between type and director
from scipy.stats import chi2_contingency


# In[29]:


# Define the variable pairs for chi-square tests
variable_pairs = [('type', 'director'), ('type', 'month_added'), ('type', 'country'), ('rating', 'country')]


# In[35]:


# Perform chi-square tests
for var1, var2 in variable_pairs:
    contingency_table = pd.crosstab(df1[var1], df1[var2])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for Independence between '{var1}' and '{var2}':")
    print(f"P-value: {p_value}")
    print("")


# In[39]:


# Define the variable pairs for chi-square tests
pairs = [('month_added', 'director'), ('director', 'rating'), ('country', 'month_added')]


# In[40]:


# Perform chi-square tests
for var1, var2 in pairs:
    contingency_table = pd.crosstab(df1[var1], df1[var2])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for Independence between '{var1}' and '{var2}':")
    print(f"P-value: {p_value}")
    print("")


# In[41]:


#a p-value higher than the significance level of 0.05 suggests independence between the two variables; there is no significant association between the two variables


# In[47]:


#Encoding:
from sklearn.preprocessing import LabelEncoder
# Initialize encoders
label_encoder = LabelEncoder()


# In[48]:


columns_to_encode = ['country', 'rating', 'type', 'title', 'director', 'date_added','listed_in','month_added']


# In[49]:


# Apply label encoding to selected columns
for column in columns_to_encode:
    df1[column] = label_encoder.fit_transform(df1[column])


# In[50]:


df1.head()


# In[51]:


correlation_matrix = df1.corr()


# In[52]:


print(correlation_matrix)


# In[53]:


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.show()

