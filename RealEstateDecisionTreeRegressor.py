#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[2]:


#load dataset
df = pd.read_csv('realtordata.csv')


# In[3]:


#view any 10 rows
df.sample(10)


# In[4]:


#shape of the data
df.shape


# In[5]:


#how many nulls are there per column
df.isna().sum()


# In[6]:


#how mny unique values in each column
dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]
pd.DataFrame(dict, index = ["unique count"]).transpose()


# In[7]:


#deal with missing values
#first check for dependence between the variables
#Chi square test for independence
#H0:Independence
#HA:Dependence
variable_pairs = [('state', 'price'), ('bed', 'acre_lot'), ('bath', 'acre_lot'), ('city', 'state'), ('acre_lot', 'house_size'), ('acre_lot', 'state')]
# Perform chi-square tests
for var1, var2 in variable_pairs:
    contingency_table = pd.crosstab(df[var1], df[var2])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for Independence between '{var1}' and '{var2}':")
    print(f"P-value: {p_value}")
    print("")


# In[8]:


#a p-value less than the significance level of 0.05 suggests dependence
#1st we replace the variables that depend on state since state doesnt have any missing values
#group data by state
stategrp = df.groupby('state')


# In[9]:


# Calculating mean price and median acre_lot per state
state_mean_price = stategrp['price'].transform('mean')
state_median_acre_lot = stategrp['acre_lot'].transform('median')


# In[10]:


df['price'].fillna(state_mean_price, inplace=True)
df['acre_lot'].fillna(state_median_acre_lot, inplace=True)


# In[11]:


#city is a string variable
# Function to find the most frequent non-null city within each state
# Function to find the most frequent non-null city within each state
def most_frequent_city(x):
    if x.notnull().any():
        return x.value_counts().idxmax()
    else:
        return np.nan  # Return NaN if all values are null

# Impute missing values in the 'city' column with the most frequent city per state
most_frequent_city_per_state = stategrp['city'].transform(most_frequent_city)
df['city'].fillna(most_frequent_city_per_state, inplace=True)


# In[12]:


df.isna().sum()


# In[13]:


#there are still 25 missing acre_lot values and 3 missing city values
# Filter rows with missing city
missing_city_rows = df[df['city'].isnull()]

# Filter rows with missing acre_lot
missing_acre_lot_rows = df[df['acre_lot'].isnull()]

# Display the rows
print("Rows with missing city:")
print(missing_city_rows)

print("\nRows with missing acre_lot:")
print(missing_acre_lot_rows)


# In[14]:


# Filter rows where the state is Louisiana
louisiana_rows = df[df['state'] == 'South Carolina']

# Display the rows
print("Rows where the state is Louisiana:")
print(louisiana_rows)


# In[15]:


# Impute missing values in 'city' column where 'state' is Louisiana with "Unknown"
df.loc[df['state'] == 'Louisiana', 'city'] = "Unknown"
# Calculate the mean of the acre_lot column
mean_acre = df['acre_lot'].mean()

# Impute missing values in 'acre_lot' column with the mean
df['acre_lot'].fillna(mean_acre, inplace=True)


# In[16]:


#the columns bed, bath and house_size are dependent on the column acre_lot
#group data by acre_lot
acregrp = df.groupby('acre_lot')


# In[17]:


# Calculating mean house_size and mode for bath and bed per acre_lot
mean_hsize = acregrp['house_size'].transform('mean')


# In[18]:


# Function to calculate the mode, handling cases where all values are null
def calculate_mode(x):
    if x.notnull().any():
        return x.mode().iloc[0]
    else:
        return np.nan  # Return NaN if all values are null
mode_bath = acregrp['bath'].transform(calculate_mode)
mode_bed = acregrp['bed'].transform(calculate_mode)


# In[19]:


# Impute missing values in 'house_size', bath and bed column
df['house_size'].fillna(mean_hsize, inplace=True)

df['bath'].fillna(mode_bath, inplace=True)

df['bed'].fillna(mode_bed, inplace=True)


# In[20]:


df.isna().sum()


# In[21]:


#group data by price
pricegrp = df.groupby('price')


# In[22]:


# Calculating mean house_size and mode for bath and bed per acre_lot
mean_hsize2 = pricegrp['house_size'].transform('mean')


# In[23]:


# Impute missing values in 'house_size', bath and bed column
df['house_size'].fillna(mean_hsize2, inplace=True)


# In[24]:


# Calculating mean price and median acre_lot per state
state_median_acre_lot = stategrp['acre_lot'].transform('median')
state_median_acre_lot = stategrp['acre_lot'].transform('median')


# In[25]:


#zipcode
df['zip_code'].fillna(12345, inplace=True)


# In[26]:


# Calculate mode of 'bed' column
mode_bed = df['bed'].mode()[0]
# Calculate mode of 'bath' column
mode_bath = df['bath'].mode()[0]
# Calculate mode of 'house_size' column
mode_house_size = df['house_size'].mode()[0]


# In[27]:


# Impute missing values in 'bed' column with mode
df['bed'].fillna(mode_bed, inplace=True)

# Impute missing values in 'bath' column with mode
df['bath'].fillna(mode_bath, inplace=True)

# Impute missing values in 'house_size' column with mode
df['house_size'].fillna(mode_house_size, inplace=True)


# In[28]:


#correlation
df.corr(method = 'spearman')


# In[29]:


#visualize correlation matrix using heatmap
correlation_matrix = df.corr(method = 'spearman')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix HeatMap')
plt.show()


# In[30]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# In[31]:


# Split the dataset into features (X) and target variable (y)
X = df[['bed', 'bath', 'house_size', 'acre_lot']]
y = df['price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


#model ---Decision Tree Regressor for regression analysis
model = DecisionTreeRegressor()


# In[33]:


# Fit the model
model.fit(X_train, y_train)


# In[34]:


# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[35]:


# Evaluate the model
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[36]:


#the model captures 92.4% of relationships


# In[38]:


# Function to get input values from the user
def get_input():
    bed = int(input("Enter number of bedrooms: "))
    bath = int(input("Enter number of bathrooms: "))
    house_size = float(input("Enter house size (in square feet): "))
    acre_lot = float(input("Enter acre lot: "))
    return bed, bath, house_size, acre_lot

# Get input values from the user
bed, bath, house_size, acre_lot = get_input()

# Predict the price using the decision tree regressor model
predicted_price = model.predict([[bed, bath, house_size, acre_lot]])

print("Predicted Price:", predicted_price[0])


# In[40]:


# Function to get input values from the user
def get_input():
    bed = int(input("Enter number of bedrooms: "))
    bath = int(input("Enter number of bathrooms: "))
    house_size = float(input("Enter house size (in square feet): "))
    acre_lot = float(input("Enter acre lot: "))
    return bed, bath, house_size, acre_lot

# Get input values from the user
bed, bath, house_size, acre_lot = get_input()

# Predict the price using the decision tree regressor model
predicted_price = model.predict([[bed, bath, house_size, acre_lot]])

print("Predicted Price:", predicted_price[0])

