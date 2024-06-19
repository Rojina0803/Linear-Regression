import numpy as np
import pandas as pd

# data must be in matrix form
#creating data

X= np.arange(30).reshape(30,1)
y=[[x[0]**2]for x in X]
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=101) #sends 20% data from both x and y for testing

from sklearn.linear_model import LinearRegression
linear_regression= LinearRegression()
linear_regression.fit(X_train,y_train)
print("Y:",y_test)
print("X:",X_test)

y_pred=linear_regression.predict(X_test)
print(y_pred)

# y_pred=linear_regression.predict([20])
# print(y_pred)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print(r2)