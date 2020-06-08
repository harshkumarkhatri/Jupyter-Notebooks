from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv("CarPrice_Assignment.csv")
print(np.size(dataset))
# print(dataset.head())
dataset.drop(columns='car_ID',inplace=True)
# print(dataset.head())
dataset.info()
print(dataset.describe().T)

# Checking for missing or inappropriate values
print((dataset.isna().sum()/dataset.shape[0])*100)
# We can also do the check by
print("Dataframe command")
print(pd.DataFrame(dataset.isnull()))

print(dataset.head())
x=dataset[['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']]
print(x.head())
y=dataset['price']
print(y.head())
from sklearn.model_selection import train_test_split
# Splitting the dataset
x_train,x_test,y_traian,y_test=train_test_split(x,y,test_size=0.33)
print(np.size(x_train))
print(np.size(y_traian))

# Plotting the training and testing data
# plt.figure(figsize=(20,15))
# plt.scatter(x_train['wheelbase'],y,c='r')
# plt.scatter(x_test,y_test,c='b')
# plt.show()

# intiating the model
reg=linear_model.LinearRegression()
reg.fit(x_train,y_traian)
y_pred=reg.predict(x_test)
print("Trainign data")
print(y_test.head())
print("Predicted")
print(y_pred)
print(reg.score(x_test,y_test))

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))