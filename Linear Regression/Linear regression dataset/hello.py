from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
dataset=pd.read_csv("Linear Regression - Sheet1.csv")
print(dataset.head())
print(np.shape(dataset))
X=dataset['X'].values
y=dataset['Y'].values
print(np.shape(X))
print(np.shape(y))
X=X.reshape(-1,1)
y=y.reshape(-1,1)
print(np.shape(X))
print(np.shape(y))
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
plt.scatter(x_train,y_train,c='y')
preds=reg.predict(x_test)
plt.plot(x_test,preds,'r',linewidth=2)
plt.show()
print("Score for training \n")
print(reg.score(x_train,y_train))
print("Score ffor testing data")
print(reg.score(x_test,y_test))