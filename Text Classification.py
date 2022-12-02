#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns


# In[85]:


spam_dataset = pd.read_csv('spam_messages.csv',encoding='latin-1') 
spam_dataset.head()


# In[86]:


spam_dataset = spam_dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
spam_dataset = spam_dataset.rename(columns={"v1":"spam_label", "v2":"messages"})
spam_dataset.head()


# In[87]:


spam_dataset.spam_label.value_counts()


# In[88]:


spam_dataset["binary_output"]=spam_dataset["spam_label"].map({'ham':1,'spam':0});


# In[89]:


spam_dataset['text_length'] = spam_dataset['messages'].apply(len)
spam_dataset.head()


# In[90]:


spam_dataset.hist(column='text_length', by='spam_label', bins=30,figsize=(20,6))


# In[91]:


X = spam_dataset['messages']
Y = spam_dataset['binary_output']
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=0)


# In[92]:


from sklearn.feature_extraction.text import CountVectorizer  #to remove stopwords like (the,they,their)
vector = CountVectorizer(stop_words ='english')
vector.fit(X_train)


# In[93]:


X_train_transformed =vector.transform(X_train)
X_test_transformed =vector.transform(X_test)


# In[94]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_transformed,Y_train)
y_pred = model.predict(X_test_transformed)
y_pred_prob = model.predict_proba(X_test_transformed)


# In[95]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))


# In[96]:


print("Precision Score - > ",precision_score(Y_test,y_pred))
print("Recall Score - > ",recall_score(Y_test,y_pred))
print("F1 Score - > ",f1_score(Y_test,y_pred))

