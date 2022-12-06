#!/usr/bin/env python
# coding: utf-8

# # GRIP:The Spark Foundation

# Data Science and Bussiness Analytics Intern

# Author: Shalaka Patil

# # Task 2 :Prediction Using Unsupervised ML

# #Importing the Data
#   In this step we will import the required libraries & data set with the help of pandas library.

# In[1]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
# To ignore the warning
import warnings as wg
wg.filterwarnings("ignore")


# In[2]:


# Reading data Iris Dataset
df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/Iris Data.csv")
print("Impoprting Data Sucessfully")


# In[3]:


df.head()


# Visualization the Data
#        In this step we will try to visualize our dataset.

# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df['Species'].unique()


# In[8]:


df.info()


# In[9]:


df.describe()


# #Now,we will drop the label column because it is an unsupervised learning problem.

# In[10]:


iris = pd.DataFrame(df)
iris_df = iris.drop(columns= ['Species' ,'Id'] )
iris_df.head( )


# # Finding the optimum no. of clusters
#     Before clustering the data using K-Means .We need to specify the number of clusters.In order to find the optimum number of clusters.There are various methods available like Elbow method.Here, the elbow method is used.

# # Brief about the Elbow Method
#     In this method,the number of clusters are various withine a certain range.For each number of within cluster sum of square value is calculated and stored in the list.These values are then plotted against the the range of number of cluster used before the location of bend in the second plot indicates the appropriate number of clusters.

# In[11]:


#Calculating the within cluster sum of square
Within_cluster_sum_of_square=[]
clusters_range=range(1,15)
for k in clusters_range:
    km=KMeans(n_clusters=k)
    km=km.fit(iris_df)
    Within_cluster_sum_of_square.append(km.inertia_)


# Plotting the Within_cluster_sum of Square against clusters range

# In[12]:


plt.plot(clusters_range,Within_cluster_sum_of_square,'go--',color='green')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within_Cluster Sum of Square')
plt.grid()
plt.show()


# We can clearly see why it is called "The elbow method" from the above graph, the optimum clusters is where the elbow occurs.This is when the within clusters sum of square dpoesn't decreases significantly with every iteration.
#    From this we choose the number of clusters as **3**

# #Applying K Means clustering on the data

# In[13]:


from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
predictions = model.fit_predict(iris_df)


# Visualising the clusters

# In[14]:


x=iris_df.iloc[:,[0,1,2,3]].values
plt.scatter(x[predictions == 0, 0], x[predictions == 0, 1],s=25,c='red',label='Iris_Setosa')
plt.scatter(x[predictions == 1, 0], x[predictions == 1, 1],s=25,c='Blue',label='Iris_Versicolour')
plt.scatter(x[predictions == 2, 0], x[predictions == 2, 1],s=25,c='green',label='Iris_Virginica')

# Plotting the cluster centers
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.legend()
plt.grid()
plt.show()

