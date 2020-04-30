# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# pip install tensorflow
# pip install --upgrade keras
# Part 1 - Data Preprocessing
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#spliting training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

