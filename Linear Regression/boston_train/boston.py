from sklearn import linear_model
import matlotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset_train=pd.read_csv("boston_train.csv")
dataset_test=pd.read_csv("boston_test.csv")
dataset_predict=pd.read_csv("boston_predict.csv")