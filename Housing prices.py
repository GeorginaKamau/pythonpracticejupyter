#!/usr/bin/env python
# coding: utf-8

# In[2]:


# working with numerical data
get_ipython().system('pip install pandas')
# working with arrays
get_ipython().system('pip install numpy')
# for visualization
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install ggplot')


# In[3]:


#Import libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12, 8)


# In[4]:


#create dataframe and store dataset there
# pd to read data
data = pd.read_csv(r'C:\Users\GKamau\Downloads\Housing.csv')


# In[5]:


#view data
data.head()


# In[6]:


#percentage of missing data using for loop
for col in data.columns:
    missinginfo = np.mean(data[col].isnull())
    print('{} - {}%' .format(col, missinginfo))


# In[8]:


#change price and area from int to float
data['price'] = data['price'].astype('float64')
data['area'] = data['area'].astype('float64')


# In[9]:


#column data types
data.dtypes


# In[10]:


#rank the houses by price 
#asc =false means in descending order -- highest price to lowest
data.sort_values(by=['price'], inplace = False, ascending = False )


# In[55]:


#descriptive statistics to summarize the central tendency, dispersion and shape of a dataset’s distribution
data.describe()


# In[11]:


#correlation of data using spearman method
data.corr(method = 'spearman')


# In[12]:


#visualize correlation matrix using heatmap
correlation_matrix = data.corr(method = 'spearman')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix HeatMap')
plt.show()


# In[16]:


#There is significant evidence to say there is a strong correlation between:
# number of stories and bedrooms
# price of houses and area -- meaning the more space (land) a house occupies the more expensive it is
# price seems to heaviluy rely on area covered by the house


# In[15]:


#regression model fit
sns.regplot(x = 'area', y = 'price', data = data, scatter_kws = {"color" : "red"}, line_kws = {"color" : "blue"})


# In[28]:


# giving unique labels to furnish status to make results easier to interpret
predct = dict(zip(data.bedrooms.unique(), data.furnishingstatus.unique()))   
predct


# In[29]:


#predict whether the house is furnished, semi furnished or unfurnished


# In[35]:


#store data in 3 different dataframes
furnished = data[data['furnishingstatus'] == 'furnished']


# In[39]:


semifurnished = data[data['furnishingstatus'] == 'semi-furnished']


# In[37]:


unfurnished = data[data['furnishingstatus'] == 'unfurnished']


# In[40]:


#contains information on furnished houses
furnished.head()


# In[41]:


unfurnished.head()


# In[42]:


semifurnished.head()


# In[50]:


furnished.plot(x='stories', y='price', kind='bar')


# In[54]:


#scatter plot of unfurnished houses according to price and area
plt.scatter(unfurnished['price'],unfurnished['area'])


# In[56]:


#scatter plot of furnished houses according to price and area
plt.scatter(furnished['price'],furnished['area'])


# In[57]:


#scatter plot of semifurnished houses according to price and area
plt.scatter(semifurnished['price'],semifurnished['area'])


# In[61]:


#use KNN to predict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[78]:


#split into train and test
x=data[['bedrooms','area','bathrooms' ]]
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[79]:


knn=KNeighborsClassifier()


# In[80]:


knn.fit(x_train,y_train)


# In[81]:


#checking accuracy of classifier
knn.score(x_test,y_test)


# In[70]:


#knn is not an accurate predictor since housing prices vary on many more factors such as economic state and developments that may sprout in the area


# In[ ]:




