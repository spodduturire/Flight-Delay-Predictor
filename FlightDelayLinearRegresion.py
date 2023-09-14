#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from AuthCredentials import client, resource


# In[ ]:


import boto3
import pandas
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

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


# In[ ]:


data2 = data
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
df = data2[data2['LateAircraftDelay'].notna()]


# In[ ]:


df = df.drop_duplicates()


# In[ ]:


df["Origin"].replace(df["Origin"].unique(), range(len(df["Origin"].unique())), inplace=True)
df["TailNum"].replace(df["TailNum"].unique(), range(len(df["TailNum"].unique())), inplace=True)
df["Dest"].replace(df["Dest"].unique(), range(len(df["Dest"].unique())), inplace=True)
df["UniqueCarrier"].replace(df["UniqueCarrier"].unique(), range(len(df["UniqueCarrier"].unique())), inplace=True)


# In[ ]:

spark = SparkSession \
        .builder \
        .master("local[1]")\
        .appName("Python Spark SQL basic example") \
        .getOrCreate()
sparkDF=spark.createDataFrame(df)
sparkDF.printSchema()
sparkDF.show()


# In[ ]:


feature_columns = sparkDF.columns[:-1]


# In[ ]:


assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
data_2 = assembler.transform(sparkDF)


# In[ ]:


train, test = data_2.randomSplit([0.7, 0.3])


# In[ ]:


algo = LinearRegression(featuresCol="features", labelCol="LateAircraftDelay")
model = algo.fit(train)


# In[ ]:

predictions = model.transform(test)
evaluator = RegressionEvaluator(
    labelCol="LateAircraftDelay", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
