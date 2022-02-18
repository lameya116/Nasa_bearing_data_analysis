import os
import pandas as pd
import numpy as np
from matplotlib import font_manager as fm, rcParams 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

filepath = os.path.dirname(__file__)
df1 = pd.read_csv(filepath+"/train.csv")
df2 = pd.read_csv(filepath+"/test.csv")

xtrain= df1[['0_FFT mean coefficient_32','rms_val','kurtosis_val','skewness','time_diff']]
ytrain = df1['peak_to_peak']
xtest= df2[['0_FFT mean coefficient_32','rms_val','kurtosis_val','skewness','time_diff']]
ytest = df2['peak_to_peak']

regr = linear_model.LinearRegression()
#polynomial_features= PolynomialFeatures(degree=1)
#xtrain = polynomial_features.fit_transform(xtrain)
regr.fit(xtrain, ytrain)
zp = regr.predict(xtest)
plt.plot(df2['time_diff'], ytest, label = "Actual", color='m')
plt.plot(df2['time_diff'], zp, label = "Linear", color='y')

sc=StandardScaler()

scaler = sc.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest= scaler.transform(xtest)
#print(ytest)
'''xtrain = xtrain.reshape((len(xtrain),1))
ytrain = ytrain.reshape((len(ytrain),1))
xtest = xtest.reshape((len(xtest),1))
ytest = ytest.reshape((len(ytest),1))'''
#t = t.reshape(len(t),1)
'''from sklearn.preprocessing import StandardScaler
sc_S = StandardScaler()
sc_t = StandardScaler()
xtrain = sc_S.fit_transform(xtrain)
ytrain = sc_t.fit_transform(ytrain)'''

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf',C=1000, gamma=0.005, epsilon=0.05)
regressor.fit(xtrain, ytrain)
y_test = regressor.predict(xtest)
#print(y_test)
plt.plot(df2['time_diff'], y_test, label = "SVR", color='r')


plt.ylabel('amplitude/s^2')
plt.xlabel('time/s')

from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(xtrain, ytrain)
y_test = rf.predict(xtest)
plt.plot(df2['time_diff'], y_test, label = "RBF", color='b')

mlp_reg = MLPRegressor(hidden_layer_sizes=(150,100,50),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(xtrain, ytrain)
y_test = rf.predict(xtest)
plt.plot(df2['time_diff'], y_test, label = "MLP", color='g')


# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
