'''
Created on Jun 10, 2017

MACHINE LEARNING - LINEAR REGRESSION PREDICTION FOR X-VALUES
- This project will predict the body weights in given brain weights 

@author: Diaz Agasatya
'''
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt') #pd = calling the pandas dependency to use the method read_fwf("whatever data file you want to use")
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression() #create variable to store the model LinearRegression derived from linear_model sklearn
body_reg.fit(x_values,y_values)

#visualize results
plt.scatter(x_values,y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
