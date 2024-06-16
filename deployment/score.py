#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pickle
import pandas as pd
import numpy as np


# In[13]:


with open('./utils/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[14]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[15]:


year = input('enter the year: ')
month = input('enter the month number: ')
df = read_data(f'./data/yellow_tripdata_{year}-{month}.parquet')


# In[16]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[17]:


print(np.mean(y_pred))

