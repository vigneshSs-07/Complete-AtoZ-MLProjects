# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
#dataset=pd.read_csv('Position_Salaries.csv') 
dataset=pd.read_csv('Position_Salaries.csv')
#dataset = pd.read_csv('C:\Users\Vignesh_Ss\Desktop\Testcase\Position_Salaries.csv')
dataset.head(3)
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#import linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


#import polynominal regression
"""from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)

#visualisation of linear regression
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Truth vs Bluf')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


#visualisation of polynominal regression
import numpy as np
X_grid=np.arange(min(X), max(X), 0.1)
Y_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color='red')
plt.plot(Y_grid,lin_reg_2.predict(poly_reg.fit_transform(Y_grid)), color='blue')
plt.title('Truth vs Bluf')
plt.xlabel('X-vector')
plt.ylabel('Y-vector')
plt.show()


#prediciting the values of single value (i.e, 6.5)
lin_reg.predict([[6.5]])


lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


''' Expected 2D array, got 1D array instead:
array=[6.5].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'''
#lin_reg.predict(6.5)