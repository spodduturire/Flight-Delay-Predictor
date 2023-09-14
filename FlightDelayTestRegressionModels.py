#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from AuthCredentials import client, resource


# In[ ]:


import boto3
import pandas
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import time
import matplotlib.pyplot as plt

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
        .master("local[4]")\
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
train_time = []
eval_time = []
RMSE = []

# In[ ]:

start = time.time()
algo = LinearRegression(featuresCol="features", labelCol="LateAircraftDelay")
model = algo.fit(train)
print("Time to train LinearRegression Regression model - ")
train_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

start = time.time()
predictions = model.transform(test)
print("Evaluate LinearRegression Regression model - ")
evaluator = RegressionEvaluator(
    labelCol="LateAircraftDelay", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
RMSE.append(rmse)
eval_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

# In[ ]:

start = time.time()
algo2 = DecisionTreeRegressor(featuresCol="features", labelCol="LateAircraftDelay")
model2 = algo2.fit(train)
print("Time to train DecisionTree Regression model - ")
train_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

start = time.time()
predictions2 = model2.transform(test)
print("Evaluate DecisionTree Regression model - ")
evaluator2 = RegressionEvaluator(
    labelCol="LateAircraftDelay", predictionCol="prediction", metricName="rmse")
rmse2 = evaluator2.evaluate(predictions2)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse2)
RMSE.append(rmse2)
eval_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

# In[ ]:

start = time.time()
algo3 = RandomForestRegressor(featuresCol="features", labelCol="LateAircraftDelay")
model3 = algo3.fit(train)
print("Time to train RandomForest Regression model - ")
train_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

start = time.time()
predictions3 = model3.transform(test)
print("Evaluate RandomForest Regression model - ")
evaluator3 = RegressionEvaluator(
    labelCol="LateAircraftDelay", predictionCol="prediction", metricName="rmse")
rmse3 = evaluator3.evaluate(predictions3)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse3)
RMSE.append(rmse3)
eval_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

# In[ ]:

start = time.time()
algo4 = GBTRegressor(featuresCol="features", labelCol="LateAircraftDelay")
model4 = algo4.fit(train)
print("Time to train GBTRegressor model - ")
train_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

start = time.time()
predictions4 = model4.transform(test)
print("Evaluate GBTRegressor model - ")
evaluator4 = RegressionEvaluator(
    labelCol="LateAircraftDelay", predictionCol="prediction", metricName="rmse")
rmse4 = evaluator4.evaluate(predictions4)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse4)
RMSE.append(rmse4)
eval_time.append(time.time() - start)
print(f'Time: {time.time() - start}')

# In[ ]:

x1 = ['LinearRegression','DecisionTree','RandomForest','GBTRegressor']
plt.plot(x1, train_time, label = "Train Time")
plt.plot(x1, eval_time, label = "Evaluation Time")

plt.xlabel('Regression Model')
plt.ylabel('Train Time')
plt.title('Regression Model vs Train Time')
plt.legend()
plt.show()

# In[ ]:

x2 = ['LinearRegression','DecisionTree','RandomForest','GBTRegressor']
plt.plot(x2, RMSE, label = "RMSE")

plt.xlabel('Regression Model')
plt.ylabel('RootMeanSquaredError')
plt.title('Regression Model vs RootMeanSquaredError')
plt.legend()
plt.show()
