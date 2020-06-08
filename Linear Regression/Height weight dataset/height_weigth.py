from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("data.csv")

# print("Data is ")
# print(dataset)
# Plotting the data set
# plt.scatter(dataset['Height'],dataset['Weight'])
# plt.scatter(x_test,y_test,'blue')
# plt.show()

# Splitting the data set into training and testing
x_train,x_test,y_train,y_test=train_test_split(dataset[['Height']],dataset[['Weight']],test_size=0.33)

# print(x_test)
# print(x_train)

# Reshaping the data
print(np.shape(x_train))
x_train.values.reshape(1,-1)
print(np.shape(x_train))
print(np.shape(y_train))
y_train.values.reshape(1,-1)
print(np.shape(y_train))

# Loading the model
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print("Score is ")
print(reg.score(dataset[['Height']],dataset[['Weight']]))

# Reshaping the testing data
x_test.values.reshape(1,-1)
y_test.values.reshape(1,-1)
plt.scatter(x_test,y_test,c='r')

# Finding the error
mae_sum=0
# print(y_test)

# Predicting
preds=reg.predict(x_test)

# print("Predictions are")
# print(preds)
# print("Original vlaues are")
# print(y_test)

# Plotting the testing data and the output data
plt.scatter(x_test,preds,c='b')
for b in y_test.values:
    mae_sum+=b
    # print(mae_sum)
mae_sum1=0
for a in preds:
    mae_sum1+=a
    # print(mae_sum1)

# print("Error is ")
# print((mae_sum1-mae_sum)/mae_sum)
# plt.plot(x_test,preds,'r',linewidth=2)

plt.show()