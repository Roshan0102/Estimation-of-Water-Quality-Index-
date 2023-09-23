## WATER QUALITY PREDICTION ##
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TN WATER.csv')
print(df.head())

df.shape

cols = ['PH', 'EC', 'TH', 'CA', 'MG', 'NA', 'HCO3', 'CL', 'SO4', 'NO3']
X = df[cols]
y = df['WQI']

"""# Data Visualization


"""

corr = df.corr()
corr

import seaborn as sns
plt.figure(figsize=(16,8))
sns.heatmap(corr,cmap='rainbow',annot=True)

plt.show()

sns.distplot(y,color = 'm')

# import seaborn as sns
# sns.pairplot(df)
# plt.show()

"""#NORMALIZATION#"""

X_scaled = X.copy()
for column in X_scaled.columns:
    X_scaled[column] = X_scaled[column]  / X_scaled[column].abs().max()

X_scaled

"""### Train_Test_Split"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.1, random_state = 252)

np.set_printoptions(suppress=True)

print("x_train:", X_train.shape)
print("y_train:", y_train.shape)
print("x_test:", X_test.shape)
print("y_test:", y_test.shape)

"""### Linear Regression"""

from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
model_lin = LinearRegression()
model_lin.fit(X_train, y_train)
y_pred_lin = model_lin.predict(X_test)
mse_lin = mse(y_test, y_pred_lin)
print("MSE:", f'{mse_lin:.20f}')
rmse_lin = mse(y_test, y_pred_lin, squared = False)
print("RMSE:", f'{rmse_lin:.10f}')
r2_lin = r2_score(y_test,y_pred_lin)
print('R2 Score: ',r2_lin )
MAE_lin = mae(y_test,y_pred_lin)
print('Mean Absolute Error :', f'{MAE_lin:.10f}')
MAPE_lin = mape(y_test, y_pred_lin)
print('Mean Absolute Percentage Error :',f'{MAPE_lin:.10f}')
import numpy as np
MDAPE_lin = np.median((np.abs(np.subtract(y_test,y_pred_lin)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_lin:.10f}')
NSE_lin = (1-(np.sum((y_pred_lin-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_lin)

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_lin, label = "Pred", marker = "o", color = "red")
plt.title('Multiple Linear Regressor Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_lin, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.title('Multiple Linear Regression Model', fontsize=14)
plt.show()

"""### Support Vector Regressor"""

from sklearn.svm import SVR
model_svr = SVR(kernel='poly',gamma='scale',C= 5.0)
model_svr.fit(X_train, y_train)
y_pred_svr = model_svr.predict(X_test)
mse_svr = mse(y_test, y_pred_svr)
print("MSE:", f'{mse_svr:.10f}')
rmse_svr = mse(y_test, y_pred_svr, squared = False)
print("RMSE:", f'{rmse_svr:.10f}')
r2_svr = r2_score(y_test,y_pred_svr)
print('R2 Score: ',r2_svr )
MAE_svr = mae(y_test,y_pred_svr)
print('Mean Absolute Error :', f'{MAE_svr:.10f}')
MAPE_svr = mape(y_test, y_pred_svr)
print('Mean Absolute Percentage Error :',f'{MAPE_svr:.10f}')
MDAPE_svr = np.median((np.abs(np.subtract(y_test,y_pred_svr)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_svr:.10f}')
NSE_svr = (1-(np.sum((y_pred_svr-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_svr)
# y_pred = model_svr.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_svr, label = "Pred", marker = "o", color = "red")
plt.title('SVR Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_svr, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('SVR Model', fontsize=14)
plt.legend()
plt.show()

"""### XGBOOST Regresssor"""

import xgboost as xg
model_xgb = xg.XGBRegressor(colsample_bytree= 0.7,
 learning_rate= 0.07,
 max_depth= 5,
 min_child_weight= 4,
 n_estimators= 500,
 nthread= 4,
 objective= 'reg:linear',
 silent= 1,
 subsample= 0.7)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mse(y_test, y_pred_xgb)
print("MSE:", f'{mse_xgb:.10f}')
rmse_xgb = mse(y_test, y_pred_xgb, squared = False)
print("RMSE:", f'{rmse_xgb:.10f}')
r2_xgb = r2_score(y_test,y_pred_xgb)
print('R2 Score: ',r2_xgb )
MAE_xgb = mae(y_test,y_pred_xgb)
print('Mean Absolute Error :', f'{MAE_xgb:.10f}')
MAPE_xgb = mape(y_test, y_pred_xgb)
print('Mean Absolute Percentage Error :',f'{MAPE_xgb:.10f}')
MDAPE_xgb = np.median((np.abs(np.subtract(y_test,y_pred_xgb)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_xgb:.10f}')
NSE_xgb = (1-(np.sum((y_pred_xgb-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_xgb)
# y_pred = model_xgb.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_xgb, label = "Pred", marker = "o", color = "red")
plt.title('XGBOOST Regressor Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_xgb, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('XGB Regression Model', fontsize=14)
plt.legend()
plt.show()

"""### Least Angle Regressor"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Lars
model_LAR = Lars()
model_LAR.fit(X_train, y_train)
y_pred_LAR = model_LAR.predict(X_test)
mse_LAR = mse(y_test, y_pred_LAR)
print("MSE:", f'{mse_LAR:.10f}')
rmse_LAR = mse(y_test, y_pred_LAR, squared = False)
print("RMSE:", f'{rmse_LAR:.10f}')
r2_LAR = r2_score(y_test,y_pred_LAR)
print('R2 Score: ',r2_LAR )
MAE_LAR = mae(y_test,y_pred_LAR)
print('Mean Absolute Error :', f'{MAE_LAR:.10f}')
MAPE_LAR = mape(y_test, y_pred_LAR)
print('Mean Absolute Percentage Error :',f'{MAPE_LAR:.10f}')
MDAPE_LAR = np.median((np.abs(np.subtract(y_test,y_pred_LAR)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_LAR:.10f}')
NSE_LAR = (1-(np.sum((y_pred_LAR-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_LAR)
# y_pred = model_LAR.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_LAR, label = "Pred", marker = "o", color = "red")
plt.title('LARS Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_LAR, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('LARS Regression Model', fontsize=14)
plt.show()

"""### Bayesian Ridge Regressor


"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import BayesianRidge
model_BR = BayesianRidge()
model_BR.fit(X_train, y_train)
y_pred_BR = model_BR.predict(X_test)
mse_BR = mse(y_test, y_pred_BR)
print("MSE:", f'{mse_BR:.10f}')
rmse_BR = mse(y_test, y_pred_BR, squared = False)
print("RMSE:", f'{rmse_BR:.10f}')
r2_BR = r2_score(y_test,y_pred_BR)
print('R2 Score: ',r2_BR )
MAE_BR = mae(y_test,y_pred_BR)
print('Mean Absolute Error :', f'{MAE_BR:.10f}')
MAPE_BR = mape(y_test, y_pred_BR)
print('Mean Absolute Percentage Error :',f'{MAPE_BR:.10f}')
MDAPE_BR = np.median((np.abs(np.subtract(y_test,y_pred_BR)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_BR:.10f}')
NSE_BR = (1-(np.sum((y_pred_BR-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_BR)
# y_pred = model_BR.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_BR, label = "Pred", marker = "o", color = "red")
plt.title('Bayesian Ridge Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_BR, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('BR Regression Model', fontsize=14)
plt.legend()
plt.show()

"""### Random Forest Regressor"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
model_RF = RandomForestRegressor(bootstrap= True,
 max_features= 'auto',
 min_samples_split= 2,
 n_estimators= 100)
model_RF.fit(X_train, y_train)
y_pred_RF = model_RF.predict(X_test)
mse_RF = mse(y_test, y_pred_RF)
print("MSE:", f'{mse_RF:.10f}')
rmse_RF = mse(y_test, y_pred_RF, squared = False)
print("RMSE:", f'{rmse_RF:.10f}')
r2_RF = r2_score(y_test,y_pred_RF)
print('R2 Score: ',r2_RF )
MAE_RF = mae(y_test,y_pred_RF)
print('Mean Absolute Error :', f'{MAE_RF:.10f}')
MAPE_RF = mape(y_test, y_pred_RF)
print('Mean Absolute Percentage Error :',f'{MAPE_RF:.10f}')
MDAPE_RF = np.median((np.abs(np.subtract(y_test,y_pred_RF)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_RF:.10f}')
NSE_RF = (1-(np.sum((y_pred_RF-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_RF)
# y_pred = model_RF.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_RF, label = "Pred", marker = "o", color = "red")
plt.title('Random Forest Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_RF, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('RF Regression Model', fontsize=14)
plt.legend()
plt.show()

"""### ElasticNet Regressor"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import ElasticNet
model_EN = ElasticNet(max_iter = 2000,random_state=0)
model_EN.fit(X_train,y_train)
y_pred_EN = model_EN.predict(X_test)
mse_EN = mse(y_test, y_pred_EN)
print("MSE:", f'{mse_EN:.10f}')
rmse_EN = mse(y_test, y_pred_EN, squared = False)
print("RMSE:", f'{rmse_EN:.10f}')
r2_EN = r2_score(y_test,y_pred_EN)
print('R2 Score: ',r2_EN )
MAE_EN = mae(y_test,y_pred_EN)
print('Mean Absolute Error :', f'{MAE_EN:.10f}')
MAPE_EN = mape(y_test, y_pred_EN)
print('Mean Absolute Percentage Error :',f'{MAPE_EN:.10f}')
MDAPE_EN = np.median((np.abs(np.subtract(y_test,y_pred_EN)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_EN:.10f}')
NSE_EN = (1-(np.sum((y_pred_EN-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_EN)
# y_pred = model_EN.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_EN, label = "Pred", marker = "o", color = "red")
plt.title('ElasticNet Model')
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_EN, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('EN Regression Model', fontsize=14)
plt.legend()
plt.show()

"""# Comaparison of Model Results"""

print("LR RMSE:", f'{rmse_lin:.20f}')
print("LAR RMSE:", f'{rmse_LAR:.20f}')
print("BR RMSE:", f'{rmse_BR:.20f}')

print("LR RMSE:", f'{rmse_lin}')
print("LAR RMSE:", f'{rmse_LAR}')
print("BR RMSE:", f'{rmse_BR}')

MAE_lin = mae(y_test,y_pred_lin)
print('Mean Absolute Error :', f'{MAE_lin:.20f}')
MAE_LAR = mae(y_test,y_pred_LAR)
print('Mean Absolute Error :', f'{MAE_LAR:.20f}')
MAE_LAR = mae(y_test,y_pred_LAR)
print('Mean Absolute Error :', f'{MAE_LAR:.20f}')

MAPE_lin = mape(y_test, y_pred_lin)
print('Mean Absolute Percentage Error :',f'{MAPE_lin:.20f}')
MAPE_LAR = mape(y_test, y_pred_LAR)
print('Mean Absolute Percentage Error :',f'{MAPE_LAR:.20f}')
MAPE_BR = mape(y_test, y_pred_BR)
print('Mean Absolute Percentage Error :',f'{MAPE_BR:.20f}')

mse_lin = mse(y_test, y_pred_lin)
print("MSE:", f'{mse_lin:.25f}')
mse_LAR = mse(y_test, y_pred_LAR)
print("MSE:", f'{mse_LAR:.25f}')
mse_BR = mse(y_test, y_pred_BR)
print("MSE:", f'{mse_BR:.25f}')

MDAPE_lin = np.median((np.abs(np.subtract(y_test,y_pred_lin)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_lin:.20f}')
MDAPE_LAR = np.median((np.abs(np.subtract(y_test,y_pred_LAR)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_LAR:.20f}')
MDAPE_BR = np.median((np.abs(np.subtract(y_test,y_pred_BR)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_BR:.20f}')

import matplotlib.pyplot as plt
import seaborn as sns
RMSE = []
RMSE.append(round(rmse_lin,2))
RMSE.append(round(rmse_svr,2))
RMSE.append(round(rmse_xgb,2))
RMSE.append(round(rmse_EN,2))
RMSE.append(round(rmse_RF,2))
RMSE.append(round(rmse_BR,2))
RMSE.append(round(rmse_LAR,2))
print(RMSE)
x = ["LR","SVR","XGB","EN","RF","BR","LAR"]

def addlabels(x,RMSE):
    for i in range(len(x)):
        plt.text(i,RMSE[i],RMSE[i])
plt.figure(figsize=(15,6))
plt.bar(x,RMSE, color = "red")
addlabels(x,RMSE)
plt.xlabel("Models")
plt.ylabel("RMSE value")
plt.title("RMSE COMPARISON")
plt.show()

RMSE1 = []
RMSE1.append(round(rmse_lin,16))
# RMSE1.append(round(rmse_mlp,2))
RMSE1.append(round(rmse_BR,16))
RMSE1.append(round(rmse_LAR,16))
print(RMSE1)
x = ["LR","BR","LAR"]

def addlabels(x,RMSE1):
    for i in range(len(x)):
        plt.text(i,RMSE1[i],RMSE1[i])

plt.bar(x,RMSE1, color = "red")
addlabels(x,RMSE1)
plt.xlabel("Models")
plt.ylabel("RMSE value")
plt.title("Models performance")
plt.show()

"""## ENSEMBLE Model"""

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Lars, BayesianRidge
from sklearn.model_selection import train_test_split
model_VR = VotingRegressor([('lr', LinearRegression()), ('lar', Lars()), ('brr', BayesianRidge())])
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y, test_size = 0.1, random_state = 50)
model_VR.fit(X_train2, y_train2)
y_pred_VR = model_VR.predict(X_test2)
mse_VR = mse(y_test2, y_pred_VR)
print("MSE:", f'{mse_VR:.10f}')
rmse_VR = mse(y_test2, y_pred_VR, squared = False)
print("RMSE:", f'{rmse_VR:.10f}')
r2_VR = r2_score(y_test2,y_pred_VR)
print('R2 Score: ',r2_VR)
MAE_VR = mae(y_test2,y_pred_VR)
print('Mean Absolute Error :', f'{MAE_VR:.10f}')
MAPE_VR = mape(y_test2, y_pred_VR)
print('Mean Absolute Percentage Error :',f'{MAPE_VR:.10f}')
MDAPE_VR = np.median((np.abs(np.subtract(y_test2,y_pred_VR)/ y_test2))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_VR:.10f}')
NSE_VR = (1-(np.sum((y_pred_VR-y_test2)**2)/np.sum((y_test2-np.mean(y_test2))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_VR)

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test2, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_VR, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test2, y=y_pred_VR, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('ENSEMLE Model', fontsize=14)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
RMSE1 = []
RMSE1.append(round(rmse_BR,16))
RMSE1.append(round(rmse_VR,16))

print(RMSE1)
x = ["BR","Hybrid"]

def addlabels(x,RMSE1):
    for i in range(len(x)):
        plt.text(i,RMSE1[i],RMSE1[i])
plt.figure(figsize=(15,6))
plt.bar(x,RMSE1, color = "red")
addlabels(x,RMSE1)
plt.xlabel("Models")
plt.ylabel("RMSE value")
plt.title("RMSE COMPARISON")
plt.show()

"""## PCA USED MODELS ##"""

from sklearn.decomposition import PCA

pca=PCA(n_components = 9)

pca.fit(X_scaled)

X_pca=pca.transform(X_scaled)

X_pca.shape

X_pca

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=df['WQI'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pca, y, test_size = 0.1, random_state = 42)

np.set_printoptions(suppress=True)

"""### Linear Regressor(PCA)"""

from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
model_pca_lin = LinearRegression()
model_pca_lin.fit(X_train1, y_train1)
y_pred_pca_lin = model_pca_lin.predict(X_test1)
mse_pca_lin = mse(y_test1, y_pred_pca_lin)
print("MSE:", f'{mse_pca_lin:.10f}')
rmse_pca_lin = mse(y_test1, y_pred_pca_lin, squared = False)7
print("RMSE:", f'{rmse_pca_lin:.10f}')
r2_pca_lin = r2_score(y_test1,y_pred_pca_lin)
print('R2 Score: ',r2_pca_lin )
MAE_pca_lin = mae(y_test1,y_pred_pca_lin)
print('Mean Absolute Error :', f'{MAE_pca_lin:.10f}')
MAPE_pca_lin = mape(y_test1, y_pred_pca_lin)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_lin:.10f}')
MDAPE_pca_lin = np.median((np.abs(np.subtract(y_test1,y_pred_pca_lin)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_lin:.10f}')
NSE_pca_lin = (1-(np.sum((y_pred_pca_lin-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_lin)
# y_pred = model_lin.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')
# print(rmse_lin)

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_lin, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_lin, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('Linear Regression Model', fontsize=14)
plt.legend()
plt.show()

"""### Support Vector Regressor(PCA)"""

from sklearn.svm import SVR
model_pca_svr = SVR(kernel='poly',gamma = 'scale',C=5.0)
model_pca_svr.fit(X_train1, y_train1)
y_pred_pca_svr = model_pca_svr.predict(X_test1)
mse_pca_svr = mse(y_test1, y_pred_pca_svr)
print("MSE:", f'{mse_pca_svr:.10f}')
rmse_pca_svr = mse(y_test1, y_pred_pca_svr, squared = False)
print("RMSE:", f'{rmse_pca_svr:.10f}')
r2_pca_svr = r2_score(y_test1,y_pred_pca_svr)
print('R2 Score: ',r2_pca_svr )
MAE_pca_svr = mae(y_test1,y_pred_pca_svr)
print('Mean Absolute Error :', f'{MAE_pca_svr:.10f}')
MAPE_pca_svr = mape(y_test1, y_pred_pca_svr)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_svr:.10f}')
MDAPE_pca_svr = np.median((np.abs(np.subtract(y_test1,y_pred_pca_svr)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_svr:.10f}')
NSE_pca_svr = (1-(np.sum((y_pred_pca_svr-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_svr)
# y_pred = model_svr.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_svr, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_svr, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('SVRegression Model', fontsize=14)
plt.legend()
plt.show()

"""### XGBOOST Regressor(PCA)"""

import xgboost as xg
model_pca_xgb = xg.XGBRegressor(colsample_bytree= 0.7,
 learning_rate= 0.07,
 max_depth= 5,
 min_child_weight= 4,
 n_estimators= 500,
 nthread= 4,
 objective= 'reg:linear',
 silent= 1,
 subsample= 0.7)
model_pca_xgb.fit(X_train1, y_train1)
y_pred_pca_xgb = model_pca_xgb.predict(X_test1)
mse_pca_xgb = mse(y_test1, y_pred_pca_xgb)
print("MSE:", f'{mse_pca_xgb:.10f}')
rmse_pca_xgb = mse(y_test1, y_pred_pca_xgb, squared = False)
print("RMSE:", f'{rmse_pca_xgb:.10f}')
r2_pca_xgb = r2_score(y_test1,y_pred_pca_xgb)
print('R2 Score: ',r2_pca_xgb )
MAE_pca_xgb = mae(y_test1,y_pred_pca_xgb)
print('Mean Absolute Error :', f'{MAE_pca_xgb:.10f}')
MAPE_pca_xgb = mape(y_test1, y_pred_pca_xgb)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_xgb:.10f}')
MDAPE_pca_xgb = np.median((np.abs(np.subtract(y_test1,y_pred_pca_xgb)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_xgb:.10f}')
NSE_pca_xgb = (1-(np.sum((y_pred_pca_xgb-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_xgb)
# y_pred = model_xgb.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_xgb, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_xgb, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('xgb Regression Model', fontsize=14)
plt.legend()
plt.show()

"""### Least Angle Regressor(PCA)"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Lars
model_pca_LAR = Lars()
model_pca_LAR.fit(X_train1, y_train1)
y_pred_pca_LAR = model_pca_LAR.predict(X_test1)
mse_pca_LAR = mse(y_test1, y_pred_pca_LAR)
print("MSE:", f'{mse_pca_LAR:.10f}')
rmse_pca_LAR = mse(y_test1, y_pred_pca_LAR, squared = False)
print("RMSE:", f'{rmse_pca_LAR:.10f}')
r2_pca_LAR = r2_score(y_test1,y_pred_pca_LAR)
print('R2 Score: ',r2_pca_LAR)
MAE_pca_LAR = mae(y_test1,y_pred_pca_LAR)
print('Mean Absolute Error :', f'{MAE_pca_LAR:.10f}')
MAPE_pca_LAR = mape(y_test1, y_pred_pca_LAR)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_LAR:.10f}')
MDAPE_pca_LAR = np.median((np.abs(np.subtract(y_test1,y_pred_pca_LAR)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_LAR:.10f}')
NSE_pca_LAR = (1-(np.sum((y_pred_pca_LAR-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_LAR)
# y_pred = model_LAR.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_LAR, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_LAR, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('LARS Model', fontsize=14)
plt.legend()
plt.show()

"""### Bayesian Ridge Regressor(PCA)"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import BayesianRidge
model_pca_BR = BayesianRidge()
model_pca_BR.fit(X_train1, y_train1)
y_pred_pca_BR = model_pca_BR.predict(X_test1)
mse_pca_BR = mse(y_test1, y_pred_pca_BR)
print("MSE:", f'{mse_pca_BR:.10f}')
rmse_pca_BR = mse(y_test1, y_pred_pca_BR, squared = False)
print("RMSE:", f'{rmse_pca_BR:.10f}')
r2_pca_BR = r2_score(y_test1,y_pred_pca_BR)
print('R2 Score: ',r2_pca_BR)
MAE_pca_BR = mae(y_test1,y_pred_pca_BR)
print('Mean Absolute Error :', f'{MAE_pca_BR:.10f}')
MAPE_pca_BR = mape(y_test1, y_pred_pca_BR)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_BR:.10f}')
MDAPE_pca_BR = np.median((np.abs(np.subtract(y_test1,y_pred_pca_BR)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_BR:.10f}')
NSE_pca_BR = (1-(np.sum((y_pred_pca_BR-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_BR)
# y_pred = model_BR.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_BR, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_BR, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('BR Model', fontsize=14)
plt.legend()
plt.show()

"""### Random Forest Regressor(PCA)"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
model_pca_RF = RandomForestRegressor(bootstrap= True,
 max_features= 'auto',
 min_samples_split= 2,
 n_estimators= 100)
model_pca_RF.fit(X_train1, y_train1)
y_pred_pca_RF = model_pca_RF.predict(X_test1)
mse_pca_RF = mse(y_test1, y_pred_pca_RF)
print("MSE:", f'{mse_pca_RF:.10f}')
rmse_pca_RF = mse(y_test1, y_pred_pca_RF, squared = False)
print("RMSE:", f'{rmse_pca_RF:.10f}')
r2_pca_RF = r2_score(y_test1,y_pred_pca_RF)
print('R2 Score: ',r2_pca_RF)
MAE_pca_RF = mae(y_test1,y_pred_pca_RF)
print('Mean Absolute Error :', f'{MAE_pca_RF:.10f}')
MAPE_pca_RF = mape(y_test1, y_pred_pca_RF)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_RF:.10f}')
MDAPE_pca_RF = np.median((np.abs(np.subtract(y_test1,y_pred_pca_RF)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_RF:.10f}')
NSE_pca_RF = (1-(np.sum((y_pred_pca_RF-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_RF)
# y_pred = model_RF.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_RF, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_RF, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('RF Model', fontsize=14)
plt.legend()
plt.show()

"""### Elastic Net Regressor(PCA)"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import ElasticNet
model_pca_EN = ElasticNet(selection='random',random_state=0)
model_pca_EN.fit(X_train1,y_train1)
y_pred_pca_EN = model_pca_EN.predict(X_test1)
mse_pca_EN = mse(y_test1, y_pred_pca_EN)
print("MSE:", f'{mse_pca_EN:.10f}')
rmse_pca_EN = mse(y_test1, y_pred_pca_EN, squared = False)
print("RMSE:", f'{rmse_pca_EN:.10f}')
r2_pca_EN = r2_score(y_test1,y_pred_pca_EN)
print('R2 Score: ',r2_pca_EN)
MAE_pca_EN = mae(y_test1,y_pred_pca_EN)
print('Mean Absolute Error :', f'{MAE_pca_EN:.10f}')
MAPE_pca_EN = mape(y_test1, y_pred_pca_EN)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_EN:.10f}')
MDAPE_pca_EN = np.median((np.abs(np.subtract(y_test1,y_pred_pca_EN)/ y_test1))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_pca_EN:.10f}')
NSE_pca_EN = (1-(np.sum((y_pred_pca_EN-y_test1)**2)/np.sum((y_test1-np.mean(y_test1))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_pca_EN)
# y_pred = model_EN.predict(X_train1)
# rmse = mse(y_train1, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test1, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_pca_EN, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test1, y=y_pred_pca_EN, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"},label = "predicted")
plt.title('EN Model', fontsize=14)
plt.legend()
plt.show()

MAE_pca_lin = mae(y_test1,y_pred_pca_lin)
print('Mean Absolute Error :', f'{MAE_pca_lin:.20f}')
MAE_pca_LAR = mae(y_test1,y_pred_pca_LAR)
print('Mean Absolute Error :', f'{MAE_pca_LAR:.20f}')
MAE_pca_BR = mae(y_test1,y_pred_pca_BR)
print('Mean Absolute Error :', f'{MAE_pca_BR:.20f}')

MAPE_pca_lin = mape(y_test1, y_pred_pca_lin)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_lin:.20f}')
MAPE_pca_LAR = mape(y_test1, y_pred_pca_LAR)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_LAR:.20f}')
MAPE_pca_BR = mape(y_test1, y_pred_pca_BR)
print('Mean Absolute Percentage Error :',f'{MAPE_pca_BR:.10f}')

print("LR RMSE:", f'{rmse_pca_lin:.20f}')
print("LAR RMSE:", f'{rmse_pca_LAR:.20f}')
print("BR RMSE:", f'{rmse_pca_BR:.20f}')

r2_pca_lin = r2_score(y_test1,y_pred_pca_lin)
print('R2 Score: ',f'{r2_pca_lin:.50f}')
r2_pca_LAR = r2_score(y_test1,y_pred_pca_LAR)
print('R2 Score: ',f'{r2_pca_LAR:.50f}')
r2_pca_BR = r2_score(y_test1,y_pred_pca_BR)
print('R2 Score: ',f'{r2_pca_BR:.20f}')

mse_pca_lin = mse(y_test1, y_pred_pca_lin)
print("MSE:", f'{mse_pca_lin:.10f}')
mse_pca_LAR = mse(y_test1, y_pred_pca_LAR)
print("MSE:", f'{mse_pca_LAR:.10f}')
mse_pca_BR = mse(y_test1, y_pred_pca_BR)
print("MSE:", f'{mse_pca_BR:.10f}')

import matplotlib.pyplot as plt
import seaborn as sns
RMSE = []
RMSE.append(round(rmse_pca_lin,2))
RMSE.append(round(rmse_pca_svr,2))
RMSE.append(round(rmse_pca_xgb,2))
RMSE.append(round(rmse_pca_EN,2))
RMSE.append(round(rmse_pca_RF,2))
RMSE.append(round(rmse_pca_BR,2))
RMSE.append(round(rmse_pca_LAR,2))
print(RMSE)
x = ["LR","SVR","XGB","EN","RF","BR","LAR"]

def addlabels(x,RMSE):
    for i in range(len(x)):
        plt.text(i,RMSE[i],RMSE[i])
plt.figure(figsize=(15,6))
plt.bar(x,RMSE, color = "red")
addlabels(x,RMSE)
plt.xlabel("Models")
plt.ylabel("RMSE value")
plt.title("Models performance")
plt.show()

"""## DEEP LEARNING MODELS ##"""

from sklearn.neural_network import MLPRegressor
parametrized_model_mlp =  MLPRegressor(solver='lbfgs',alpha=0.11125,hidden_layer_sizes=(150,150),activation='identity')
parametrized_model_mlp.fit(X_train,y_train)
y_pred_mlp1 = parametrized_model_mlp.predict(X_test)
rmse_mlp1 = mse(y_test, y_pred_mlp1, squared = False)
print("RMSE:", f'{rmse_mlp1:.10f}')
r2_mlp1 = r2_score(y_test,y_pred_mlp1)
print('R2 Score: ',r2_mlp1)
MAE_mlp1 = mae(y_test,y_pred_mlp1)
print('Mean Absolute Error :', f'{MAE_mlp1:.10f}')
MAPE_mlp1 = mape(y_test, y_pred_mlp1)
print('Mean Absolute Percentage Error :',f'{MAPE_mlp1:.10f}')
MDAPE_mlp1 = np.median((np.abs(np.subtract(y_test,y_pred_mlp1)/ y_test))) * 100
print('Median Absolute Pecentage Error :', f'{MDAPE_mlp1:.10f}')
NSE_mlp1 = (1-(np.sum((y_pred_mlp1-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))
print('Nash-Sutcliff-Efficiency:',NSE_mlp1)
# y_pred = parametrized_model_mlp.predict(X_train)
# rmse = mse(y_train, y_pred, squared = False)
# print("Training Accuracy :", f'{rmse:.10f}')

import matplotlib.pyplot as plt
plt.scatter(range(0,362),y_test, label = "Actual", marker = "o", color = "black")
plt.scatter(range(0,362),y_pred_mlp1, label = "Pred", marker = "o", color = "red")
plt.legend()
plt.show()

sns.regplot(x=y_test, y=y_pred_mlp1, data=df, scatter_kws={"color": "green"}, line_kws={"color": "red"})
plt.title('MLP Regression Model', fontsize=14)
plt.show()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error
import seaborn as sb
from sklearn.metrics import mean_squared_error as mse
import math
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
import random
import numpy as np
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

NN_model1 = Sequential()

# The Input Layer :
NN_model1.add(Dense(128, kernel_initializer='normal',input_dim = X_scaled.shape[1], activation='relu'))

# The Hidden Layers :
NN_model1.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model1.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model1.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model1.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model1.summary()

# from keras.callbacks import ModelCheckpoint
# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
# # checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# checkpoint = ModelCheckpoint(checkpoint, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, initial_value_threshold=None)
# callbacks_list = [checkpoint_name]

# tf.random.set_seed(1)
# NN_model1.fit(X_strain, y, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
NN_model1.fit(X_train, y_train, epochs=205, batch_size=32, validation_split = 0.2)

# Write the testing input and output variables
score = NN_model1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

NN_model1_pred = NN_model1.predict(X_test)

rmse_NN1 = mse(y_test, NN_model1_pred, squared = False)
print("RMSE:", f'{rmse_NN1:.10f}')
r2_NN1 = r2_score(y_test,NN_model1_pred)
print('R2 Score: ',r2_NN1)
MAE_NN1 = mae(y_test,NN_model1_pred)
print('Mean Absolute Error :', f'{MAE_NN1:.10f}')
MAPE_NN1 = mape(y_test, NN_model1_pred)
print('Mean Absolute Percentage Error :',f'{MAPE_NN1:.10f}')

sns.regplot(x=y_test, y=NN_model1_pred, data=df, scatter_kws={"color": "green"}, line_kws={"color": "yellow"})
plt.title('NN1 Regression Model', fontsize=14)
plt.show()

import tensorflow
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError

from tensorflow.keras.layers import  Dropout
hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.01
# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
  model = Sequential([
    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
    Dense(1, kernel_initializer='normal', activation='linear')
  ])
  return model
# build the model
NN_model2 = build_model_using_sequential()

# loss function
msle = MeanSquaredLogarithmicError()
NN_model2.compile(
    loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'],
)
# train the model
history = NN_model2.fit(
    X_train.values,
    y_train.values,
    epochs=145,
    batch_size=64,
    validation_split=0.2,
)

NN_model2_pred = NN_model2.predict(X_test)

rmse_NN2 = mse(y_test, NN_model2_pred, squared = False)
print("RMSE:", f'{rmse_NN2:.10f}')
r2_NN2 = r2_score(y_test,NN_model2_pred)
print('R2 Score: ',r2_NN2)
MAE_NN2 = mae(y_test,NN_model2_pred)
print('Mean Absolute Error :', f'{MAE_NN2:.10f}')
MAPE_NN2 = mape(y_test, NN_model2_pred)
print('Mean Absolute Percentage Error :',f'{MAPE_NN2:.10f}')

sns.regplot(x=y_test, y=NN_model2_pred, data=df, scatter_kws={"color": "green"}, line_kws={"color": "yellow"})
plt.title('NN2 Regression Model', fontsize=14)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
RMSE = []
RMSE.append(round(rmse_lin,16))
RMSE.append(round(rmse_BR,16))
RMSE.append(round(rmse_LAR,16))
RMSE.append(round(rmse_mlp1,2))
RMSE.append(round(rmse_NN1,2))

print(RMSE)
x = ["LR","BR","LAR","MLP","NN1"]

def addlabels(x,RMSE):
    for i in range(len(x)):
        plt.text(i,RMSE[i],RMSE[i])
plt.figure(figsize=(10,6))
plt.bar(x,RMSE, color = "red")
addlabels(x,RMSE)
plt.xlabel("Models")
plt.ylabel("RMSE value")
plt.title("Models performance")
plt.show()
