import os
import pandas as pd
import numpy as np
from matplotlib import font_manager as fm, rcParams 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

filepath = os.path.dirname(__file__)
# loading csv files into panda dataframe
df1 = pd.read_csv(filepath+"/train.csv")
df2 = pd.read_csv(filepath+"/test.csv")

xtrain= df1[['0_FFT mean coefficient_32','rms_val','kurtosis_val','skewness','time_diff']]
ytrain = df1['peak_to_peak']
xtest= df2[['0_FFT mean coefficient_32','rms_val','kurtosis_val','skewness','time_diff']]
ytest = df2['peak_to_peak']

sc=StandardScaler()
scaler = sc.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest= scaler.transform(xtest)

# linear regression
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
zp = regr.predict(xtest)
plt.plot(df2['time_diff'], ytest, label = "Actual", color='m')
plt.plot(df2['time_diff'], zp, label = "Linear", color='y')


# hyper perameter tunning for SVR
'''parameters = {'kernel': ['rbf'], 'C':[0.01,0.1,1,10,100,1000],'gamma': [1e-3, 0.005,0.01,0.05, 0.1],'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1]}
svr = svm.SVR()
clf = GridSearchCV(svr, parameters, verbose=2, cv=5)
clf.fit(xtrain,ytrain)
print(clf._best_estimator)
'''

regressor = SVR(kernel = 'rbf',C=1000, gamma=0.005, epsilon=0.05)
regressor.fit(xtrain, ytrain)
y_test = regressor.predict(xtest)
plt.plot(df2['time_diff'], y_test, label = "SVR", color='r')
plt.ylabel('amplitude/s^2')
plt.xlabel('time/s')


# hyper perameter tunning for RF
'''param_grid = {
    'n_estimators': [10,20,40,50],
    'max_depth': [40,50,80,90,100],
    'max_features': ['auto', 'sqrt', None] ,
    'min_samples_leaf':1,
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

estimator = RandomForestClassifier(random_state = RSEED)
rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,cv = 5,n_iter = 10, verbose = 1, random_state=RSEED) 
rs.fit(xtrain, ytrain)
print(rs.best_params_)'''

rf = RandomForestRegressor(n_estimators = 20, min_samples_leaf=1,min_samples_split=2,max_features='auto',max_depth=80,bootstrap=True)
rf.fit(xtrain, ytrain)
y_test = rf.predict(xtest)
plt.plot(df2['time_diff'], y_test, label = "RBF", color='b')


mlp_reg = MLPRegressor(hidden_layer_sizes=(150,100,50),
                       max_iter = 100,activation = 'tanh',
                       solver = 'adam')

# hyper perameter tunning for MLP
'''param_grid = { 'max_iter' : [50, 100, 200, 300, 400, 500, 1000, 10000] , 'activation' : ['relu','tanh'], 'hidden_layer_sizes':[(100,100,50),(140,100,50),(50,50,50),(140,50,40),(150,100,50)], 'alpha':[0.01, .05, .1, .001, .5], 'max_iter':[50, 100, 200, 300, 500],}
clf = GridSearchCV(mlp_reg, param_grid, cv=5)
clf.fit(xtrain, ytrain)
print(clf._best_estimator)'''

mlp_reg.fit(xtrain, ytrain)
y_test = rf.predict(xtest)
plt.plot(df2['time_diff'], y_test, label = "MLP", color='g')

# show a legend on the plot
plt.legend()
plt.show()
