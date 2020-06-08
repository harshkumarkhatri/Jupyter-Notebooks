# This is the first problem which i slved from the kaggle dataset.
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the data from the data set
data_test=pd.read_csv("train.csv")
data_test2=pd.read_csv("test.csv")

# print("shape is ")
# print(np.shape(data_test))
# data_train_x=np.array(data_test['x'])
# data_train_y=np.array(data_test['y'])

# Finding the inappropriate value
print("the value is inappropriate")
print(data_test[data_test['x']>500])
print('illocing')
data_test.iloc[213]
data_test.drop(213,axis=0,inplace=True)

# data_train_x=data_train_x.reshape(1,-1)
# print(data_train_x)
# data_train_y=data_train_y.reshape(1,-1)
# print(data_train_y)
# print(data_train_y)
# print(np.shape(data_train_y))
# print(data_train_x)
# print(data_test.head())
# print(data_test['x'].head())
# plt.scatter(data_test['x'],data_test['y'])
# plt.xlabel("x coord")
# plt.ylabel("Y coord")
# plt.show()

# Defining the model
reg=linear_model.LinearRegression()

# data_test['x'].reshape(1,-1)
# data_test['x']=np.array(data_test['x'])
# print(data_test['x'])
# np.array(data_test['y'])
# data_test['x'].reshape(1,-1)
# print("shape is ")
# print(np.shape(data_train_x))
# data_train_x=data_train_x[:,np.newaxis,2]

# Fitting the model
reg.fit(data_test[['x']],data_test[['y']])
print("Score is ")
print(reg.score(data_test[['x']],data_test[['y']]))

# For predictions
y_prdict=reg.predict(data_test2[['x']])
print(np.shape(y_prdict))

# Plotting the output
plt.scatter(data_test['x'],data_test['y'])
plt.plot(data_test2[['x']],y_prdict,'r',linewidth=2)
plt.xlabel("x coord")
plt.ylabel("Y coord")
plt.show()

# Printing the value of coefficients
print("coeff is ")
print(reg.coef_)

# Printing the value of intercepts
print("Intercept is ")
print(reg.intercept_)
