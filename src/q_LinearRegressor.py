# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:30:59 2023

@author: jzilk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model

horizon = 6 # how many time units ahead
samplesForPrediction = 7 # number of samples used for prediction
ride = "ISS"
iAgglevel = 2
aggregationLevels = ["", "_hourly", "_weekly", "_daily"]
aggregation_level = aggregationLevels[iAgglevel]
modelName = "LinearRegressor"

filename = "WaitTimes" + aggregation_level 
data = pd.read_csv("C:\\Users\\jzilk\\Documents\\HFU\\DSML/dsml_project2/data/"+filename+".csv")

data = data.iloc[:-1,:] # remove one faulty value

print(data.is_holiday.unique())

print(data.describe())
print(data.dtypes)

#code based on https://rayheberer.medium.com/generating-lagged-pandas-columns-10397309ccaf (last access 2022-05-09)
def get_lag(df, trailing_window_size):
    df_lagged = df.copy()
    for window in range(1, trailing_window_size + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df.columns]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    return(df_lagged)

"""
y = data[[ride]]
y_lagged = get_lag(y, 1)
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_lagged[ride], y_lagged[ride+"_lag1"])
plt.show()
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
column = "weather"
data.loc[:,column] = le.fit_transform(data.loc[:,column])

if aggregation_level == "":
    data["time"] = pd.to_datetime(data.time)
    data['hour'] = [x.hour for x in data.time]
    data['minute'] = [x.minute for x in data.time]
else:    
    data["date_time"] = pd.to_datetime(data.date_time)
    data['hour'] = [x.hour for x in data.date_time]
    data['day'] = [x.day for x in data.date_time]
    data['year'] = [x.year for x in data.date_time]
    

cols = [ride,"month","temperature","weather"]
# "" -> date,time,year,month,day,day_of_week,weather,temperature,is_holiday
if aggregation_level == "":
    cols = cols + ["day","year","hour",'minute',"day_of_week","is_holiday"]
# date_time,temperature,weather,month,day_of_week,is_holiday
if aggregation_level == "_hourly":
    cols = cols + ["day","year","hour","day_of_week","is_holiday"]
#date_time,temperature,weather,month,day_of_week,is_holiday
if aggregation_level == "_daily":
    cols = cols + ["day","year","day_of_week","is_holiday"]
# date_time,temperature,weather,month, #useless: day_of_week,is_holiday
if aggregation_level == "_weekly":
    cols = cols + ["year"]
    
dataIn = data[cols]

""" START Plot the scatter plot matrix """
print("is_holiday unique:",data.is_holiday.unique())
#import seaborn as sns
# plot scatter matrix using seaborn
#sns.set_theme(style="ticks")
#sns.pairplot(dataIn ) #, hue=ride
""" END Plot the scatter plot matrix """

waitTimes = dataIn[[ride]]
waitTimesLagged = get_lag(waitTimes, horizon+samplesForPrediction-1)

X = dataIn.drop(ride,axis=1)
X = X.iloc[horizon+samplesForPrediction-1:,:] # remove rows to match with lagged row count

X = pd.concat([X, waitTimesLagged.iloc[:,horizon:]], axis=1)
y = waitTimesLagged[[ride,ride+"_lag1"]]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.5, shuffle=False)

Ytrain_lag1 = Ytrain[ride+"_lag1"]
Ytest_lag1 = Ytest[ride+"_lag1"]
Ytrain = Ytrain[ride]
Ytest = Ytest[ride]

print(waitTimesLagged.dtypes)

model = linear_model.LinearRegression()
model.fit(Xtrain, Ytrain)
Ypred = model.predict(Xtest)
Ypred_train = model.predict(Xtrain)


from sklearn.metrics import mean_absolute_error, mean_squared_error
print("aggregation level:",aggregation_level,"| horizon:",horizon,"| samples used:",samplesForPrediction)
avErr = mean_absolute_error(Ytest,Ypred)
print(avErr, modelName+" model average error on test data")
avErr = mean_absolute_error(Ytrain,Ypred_train)
print(avErr, modelName+" model average error on train data")
avErr = mean_absolute_error(waitTimesLagged[ride],waitTimesLagged[ride+"_lag1"])
print(avErr, "baseline average error on all data")
avErr = mean_absolute_error(Ytest,Ytest_lag1)
print(avErr, "baseline average error on test data")
avErr = mean_absolute_error(Ytrain,Ytrain_lag1)
print(avErr, "baseline average error on test data")


xplt =  np.arange(0, Ypred.shape[0], step=1)
plt.figure()
plt.plot(xplt, Ypred, label="pred")
plt.plot(xplt, Ytest, label="true")
plt.legend()
plt.title(modelName+" prediction on test set")
plt.show()
"""
xplt =  np.arange(0, Ypred_train.shape[0], step=1)
plt.figure()
plt.plot(xplt, Ypred_train, label="pred")
plt.plot(xplt, Ytrain, label="true")
plt.legend()
plt.title(modelName+" prediction on train set")
plt.show()

pred_err = Ytest - Ypred
base_err = Ytest - Ytrain_lag1
fig=plt.figure()
plt.plot(range(len(pred_err)),pred_err,label = "model error")
plt.plot(range(len(base_err)),base_err,label = "baseline error")
plt.legend()
plt.title(f"Error comparison ({modelName})")
fig.show()
"""

plt.figure()
plt.scatter(Ytest, Ypred)
plt.ylabel("Prediction")
plt.xlabel("Actual")
plt.show()

"""
xgb = model

plt.figure()
importance=model.get_booster().get_score(importance_type='weight')
plt.title('weight')
imp_frame = pd.DataFrame(importance.items())
plt.barh(imp_frame[0],imp_frame[1])
plt.show()

plt.figure()
importance=xgb.get_booster().get_score(importance_type='total_gain')
plt.title('total_gain')
imp_frame = pd.DataFrame(importance.items())
plt.barh(imp_frame[0],imp_frame[1])
plt.show()

plt.figure()
importance=xgb.get_booster().get_score(importance_type='gain')
plt.title('gain')
imp_frame = pd.DataFrame(importance.items())
plt.barh(imp_frame[0],imp_frame[1])
plt.show()

plt.figure()
importance=xgb.get_booster().get_score(importance_type='cover')
plt.title('cover')
imp_frame = pd.DataFrame(importance.items())
plt.barh(imp_frame[0],imp_frame[1])
plt.show()

plt.figure()
importance=xgb.get_booster().get_score(importance_type='total_cover')
plt.title('total_cover')
imp_frame = pd.DataFrame(importance.items())
plt.barh(imp_frame[0],imp_frame[1])
plt.show()
"""


from sklearn.inspection import permutation_importance
importance = permutation_importance(model, Xtest,
Ytest,n_repeats=10,random_state=42,scoring='neg_mean_absolute_error')

print(importance.keys())
print(importance.importances_mean)
print(Xtest.columns)
print(importance.importances)

import matplotlib.pyplot as plt
import numpy as np
pos = np.arange(len(importance.importances_mean))
plt.figure()
plt.bar(pos,importance.importances_mean,tick_label=Xtest.columns)
plt.xticks(fontsize=15,rotation=45,ha='right')
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()


"""
import shap
explainer = shap.Explainer(model)
shap_values = explainer(Xtest)
for col in Xtest.columns:
    shap.plots.scatter(shap_values[:, col], color=shap_values)
shap.plots.beeswarm(shap_values)
import numpy as np
global_shap=np.mean(np.abs(shap_values.values),axis=0)
import matplotlib.pyplot as plt
plt.barh(Xtest.columns,global_shap)
plt.show()

plt.figure()
shap.plots.bar(shap_values)
shap.plots.waterfall(shap_values[0], max_display=10)
shap.plots.waterfall(shap_values[1], max_display=10)
"""