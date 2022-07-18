#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Here in this ML Project we are going to classify whether a transaction is done by the actual card holder or is it fraud
# Collected the data from Kaggle and since this is a binary classification model we will use Logistic Regression for training our mode


# In[3]:


# Now importing the required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[5]:


dataframe = pd.read_csv('creditcard.csv')
dataframe.head()


# In[6]:


dataframe.shape
# Here in this dataframe, in the label column 0-Transaction is legit and 1-Fraud Transaction


# In[7]:


dataframe.info()
# You can also use dataframe.isnull().sum() to check which columns have null and which columns does not have null


# In[9]:


dataframe['Class'].value_counts()
# This is unbalanced dataset because more than 98% of the transactions are legit and it becomes difficult to train a ML model with his
# unbalanced dataset


# In[20]:


# Since this is unbalanced dataset we have to seperate the Legit and Fraudent transactions from the Class Column
Legit_Transactions = dataframe[dataframe.Class==0]
Fraudulent_Transactions = dataframe[dataframe.Class==1]


# In[21]:


Legit_Transactions.shape


# In[22]:


Fraudulent_Transactions.shape


# In[27]:


Legit_Transactions.Amount.describe()


# In[28]:


Fraudulent_Transactions.Amount.describe()


# In[30]:


# If we want to get more insight about our data we can use groupby function
dataframe.groupby('Class').mean()


# In[39]:


# So here in dataframe we have 492 fradulent transactions and 284315 legit transactions. So, we will randomly pick 492 legit 
# transactions and then train our Machine Learning Model
Legit_sample = Legit_Transactions.sample(n=492)
Legit_sample


# In[ ]:


# Now you have to merge these two datasets of Legit and Fradulent Transactions of 492 each


# In[49]:


New_dataframe =  pd.concat([Legit_sample,Fraudulent_Transactions],axis=0)
New_dataframe.head()
# We want the datapoints to be added row-wise do we made axis=0 instead of column-wise which is axis=1


# In[50]:


New_dataframe.tail()


# In[51]:


New_dataframe['Class'].value_counts()


# In[52]:


New_dataframe.groupby('Class').mean()


# In[58]:


# Now separating the Label from other columns
X = New_dataframe.drop(columns='Class',axis=1)
Y = New_dataframe['Class']


# In[61]:


print(X)
print(Y)


# In[64]:


# Now splitting the data into training and testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify=Y,random_state=2)


# In[76]:


# Now training the ML Model
#model = LogisticRegression()
model = LogisticRegression(solver='lbfgs', max_iter=100000000)
model.fit(X_train,Y_train)


# In[78]:


# Checking the accuracy of the training dataset
training_model = model.predict(X_train)
training_accuracy = accuracy_score(training_model,Y_train)
print("The accuracy of training data is {}".format(training_accuracy))


# In[79]:


# Now checking the accuracy on test data
testing_model = model.predict(X_test)
testing_accuracy = accuracy_score(testing_model,Y_test)
print("The accuracy of testing data is {}".format(testing_accuracy))

# If we get very less training data accuracy and very high accuracy for test data then the model is Underfitted
# If we get very less testing data accuracy and very high accuracy for traning data then we say that the model is Overfitted

