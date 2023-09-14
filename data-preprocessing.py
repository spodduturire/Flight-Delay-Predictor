#!/usr/bin/env python
# coding: utf-8

# In[2]:


from Flights1 import client, resource


# In[3]:


import boto3
import pandas

clientResponse = client.list_buckets()
    
# Print the bucket names one by one
print('Printing bucket names...')
for bucket in clientResponse['Buckets']:
    print(f'Bucket Name: {bucket["Name"]}')

# Create the S3 object
obj = client.get_object(
    Bucket = bucket["Name"],
    Key = 'DelayedFlights.csv'
)
    
# Read data from the S3 object
data = pandas.read_csv(obj['Body'])

# Print the data frame
print('Printing the data frame...')
print(data)


# In[4]:


data2 = data


# In[5]:


del data2["Year"]
del data2["Cancelled"]
del data2["CancellationCode"]
del data2["Diverted"]
del data2["DepTime"]
del data2["CRSDepTime"]
del data2["ArrTime"]
del data2["CRSArrTime"]
del data2["ActualElapsedTime"]
del data2["CRSElapsedTime"]
del data2["Unnamed: 0"]


# In[6]:


df = data2[data2['LateAircraftDelay'].notna()]


# In[7]:


df


# In[8]:


df["Origin"].replace(df["Origin"].unique(), range(len(df["Origin"].unique())), inplace=True)
df["TailNum"].replace(df["TailNum"].unique(), range(len(df["TailNum"].unique())), inplace=True)
df["Dest"].replace(df["Dest"].unique(), range(len(df["Dest"].unique())), inplace=True)
df["UniqueCarrier"].replace(df["UniqueCarrier"].unique(), range(len(df["UniqueCarrier"].unique())), inplace=True)

