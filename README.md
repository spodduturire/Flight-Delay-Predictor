# Flight-Delay-Predictor

The goal of this project is to develop a flight delay predicting system that tells its users the
probability of a delay for a particular airline at any given time of the year. For this purpose,
Spark ML module provided by Spark was used to train a Machine Learning model that can learn from past
trends and extrapolate them to future. Spark ML was preferred over other ML libraries such as Tensorflow
and PyTorch because - 

1) We wanted to exploit the benefits that a distributed system gives us to analyze how
increasing the number of nodes in a cluster affects our training time.

2) Spark uses RDDs that not only allow nodes to benefit from shared storage but also make
use of shared memory. RDDs allow memory to be shared across clusters and this significantly
enhances our training time as the training task is an iterative process relying on data from
previous iterations.

After training multiple different machine learning models on our dataset such as
Linear Regression, Decision Trees and Random Forests, we evaluated all these models using
the Root Mean Squared Error (RMSE) performance metric. The model outputting lowest RMSE
was chosen as the go-to for our project. 

The opensource Flights Delay dataset provided by Kaggle contains airline delays for more than
300 domestic airlines in the United States (around 2 million records of data) which provided ample
to train our model accurately.
