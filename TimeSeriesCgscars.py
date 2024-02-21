#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #data manipulation and analysis
import numpy as np #support for arrays and matrices
import matplotlib.pyplot as plt #plots and charts
import seaborn as sns #statistical data visualization
#data preprocessing libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
#modules optimization
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


df=pd.read_csv('craigslist_vehicles.csv')


# In[3]:


df.sample(10)


# In[4]:


df.isna().sum()


# In[5]:


df.shape


# In[6]:


#Make a copy of the dataset
df1 = df.copy()


# In[7]:


#county has no entries all through, description has the same entry all through
df1.drop(columns = ['county'], inplace = True)
df1.drop(columns= ['description'], inplace = True)


# In[8]:


#impute with NA; url s are unique
df1['image_url'].fillna('NA', inplace = True)


# In[9]:


#Calculate the mode of the entire posting date and removal date column
pd_mode = df1['posting_date'].mode()[0]
rd_mode = df1['removal_date'].mode()[0]

#Replace the remaining missing values in the "zipcode" column with the calculated mode
df1['posting_date'].fillna(pd_mode, inplace=True)
df1['removal_date'].fillna(rd_mode, inplace=True)


# In[10]:


#check for dependence between variables--to enable imputing
# Define the variable pairs for chi-square tests
variable_pairs = [('manufacturer', 'model'), ('year', 'posting_date'), ('year', 'condition'), ('model', 'transmission'), ('model','size'), ('model','cylinders'), ('model','type'), ('model','fuel')]


# In[11]:


# Perform chi-square tests
for var1, var2 in variable_pairs:
    contingency_table = pd.crosstab(df1[var1], df1[var2])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for Independence between '{var1}' and '{var2}':")
    print(f"P-value: {p_value}")
    print("")
#a p value > 0.05 indicates independence    


# In[12]:


# Impute missing values in 'model' with the most repeated model for each manufacturer
most_repeated_model_per_manufacturer = df1.groupby('manufacturer')['model'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['model'].fillna(most_repeated_model_per_manufacturer, inplace=True)

# Impute missing values in 'manufacturer' with the most repeated manufacturer for each model
most_repeated_manufacturer_per_model = df1.groupby('model')['manufacturer'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['manufacturer'].fillna(most_repeated_manufacturer_per_model, inplace=True)

# Impute missing values in 'year' with the most repeated year for each posting date
most_year_per_postingdate = df1.groupby('posting_date')['year'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['year'].fillna(most_year_per_postingdate, inplace=True)

# Impute missing values in 'condition' with the most repeated condition for each year
most_repeated_condition_per_year = df1.groupby('year')['condition'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['condition'].fillna(most_repeated_condition_per_year, inplace=True)


# In[13]:


# Impute missing values in 'condition' with the most repeated condition for each posting date
most_repeated_condition_per_posd = df1.groupby('posting_date')['condition'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['condition'].fillna(most_repeated_condition_per_posd, inplace=True)


# In[14]:


#Impute missing values in 'model' with the most repeated model
most_repeated_model = df1['model'].mode()[0]
df1['model'].fillna(most_repeated_model, inplace=True)


# In[15]:


#Round2 Impute missing values in 'manufacturer' with the most repeated manufacturer for each type
most_repeated_manufacturer_per_type = df1.groupby('type')['manufacturer'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['manufacturer'].fillna(most_repeated_manufacturer_per_type, inplace=True)
# Impute missing values in type with the most repeated type for each model
most_repeated_type_per_model = df1.groupby('model')['type'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['type'].fillna(most_repeated_type_per_model, inplace=True)


# In[16]:


#Impute missing values in 'manufacturer' with the most repeated manufacturer
most_repeated_mfct = df1['manufacturer'].mode()[0]
df1['manufacturer'].fillna(most_repeated_mfct, inplace=True)
#Impute missing values in 'manufacturer' with the most repeated manufacturer
most_repeated_ptcl = df1['paint_color'].mode()[0]
df1['paint_color'].fillna(most_repeated_ptcl, inplace=True)


# In[17]:


# Impute missing values in type with the most repeated fuel for each model
most_repeated_fuel_per_model = df1.groupby('model')['fuel'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['fuel'].fillna(most_repeated_fuel_per_model, inplace=True)
# Impute missing values in type with the most repeated fuel for each manufacturer
most_repeated_fuel_per_mfct = df1.groupby('manufacturer')['fuel'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['fuel'].fillna(most_repeated_fuel_per_mfct, inplace=True)


# In[18]:


# Impute missing values in type with the most repeated fuel for each model
most_repeated_size_per_model = df1.groupby('model')['size'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['size'].fillna(most_repeated_size_per_model, inplace=True)
# Impute missing values in type with the most repeated fuel for each model
most_repeated_trs_per_model = df1.groupby('model')['transmission'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['transmission'].fillna(most_repeated_trs_per_model, inplace=True)
# Impute missing values in type with the most repeated fuel for each model
most_repeated_cli_per_model = df1.groupby('model')['cylinders'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['cylinders'].fillna(most_repeated_cli_per_model, inplace=True)
# Impute missing values in type with the most repeated fuel for each model
most_repeated_ptcl_per_model = df1.groupby('model')['paint_color'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df1['paint_color'].fillna(most_repeated_ptcl_per_model, inplace=True)


# In[19]:


# Impute all the remaining columns
columns_to_impute = ['cylinders', 'odometer', 'title_status', 'transmission', 'VIN', 
                     'drive', 'size', 'type', 'lat', 'long']

# Define replacement values for each column
replacement_values = {
    'cylinders': df['cylinders'].mode()[0],
    'odometer': df['odometer'].median(),
    'title_status': df['title_status'].mode()[0],
    'transmission': df['transmission'].mode()[0],
    'VIN': 'Unknown',
    'drive': df['drive'].mode()[0],
    'size': df['size'].mode()[0],
    'type': df['type'].mode()[0],
    'lat': df['lat'].median(),
    'long': df['long'].median()
}

# Impute missing values for each column
for column in columns_to_impute:
    df1[column].fillna(replacement_values[column], inplace=True)


# In[20]:


df1.isna().sum()


# In[21]:


# Convert the date columns to datetime objects
df1['posting_date'] = pd.to_datetime(df1['posting_date'])
df1['removal_date'] = pd.to_datetime(df1['removal_date'])

# Calculate the duration the cars were listed before removal in days
df1['listing_duration'] = df1['removal_date'] - df1['posting_date']

# Print the DataFrame with the calculated duration
print(df1[['posting_date', 'removal_date', 'listing_duration']])


# In[22]:


#delete rows that are duplicated
df1.drop_duplicates(subset='id', inplace=True)


# In[23]:


#filter out the necessary columns into a new df----this new df is used for the Timeseries analysis
data = df1[['posting_date', 'removal_date', 'manufacturer', 'model', 'price', 'paint_color', 'condition', 'listing_duration']]


# In[24]:


data.shape


# In[25]:


# Convert 'posting_date' and 'removal_date' columns to datetime objects
data['posting_date'] = pd.to_datetime(data['posting_date'])
data['removal_date'] = pd.to_datetime(data['removal_date'])

# Set 'posting_date' as the index of the DataFrame
data.set_index('posting_date', inplace=True)


# In[26]:


# Resample the DataFrame to daily intervals and calculate the average price and number of listings per day
daily_avg_price = data.resample('D')['price'].mean()
daily_num_listings = data.resample('D').size()

# Create a figure and subplots
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot average price on the first y-axis
ax1.plot(daily_avg_price, color='b', marker='o')
ax1.set_xlabel('Date')
ax1.set_ylabel('Average Price', color='b')
ax1.tick_params('y', colors='b')

# Create a second y-axis for the number of listings
ax2 = ax1.twinx()
ax2.plot(daily_num_listings, color='r', marker='s')
ax2.set_ylabel('Number of Listings', color='r')
ax2.tick_params('y', colors='r')

# Add title and grid
plt.title('Average Price and Number of Listings Over Time (Daily)')
plt.grid(True)

# Show plot
plt.show()


# In[27]:


data.to_csv('CarsTSdata.csv', index= False)


# In[ ]:




