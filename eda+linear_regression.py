import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Algerian_forest_fires_dataset_UPDATE.csv', header = 1)

df.head()

df.info()

df.isnull().sum()

df[df.isnull().any(axis = 1)]

df.columns = df.columns.str.strip()

df.loc[:122, 'Region'] = 0
df.loc[122:, 'Region'] = 1

df['Region'] = df['Region'].astype(int)

df = df.dropna().reset_index(drop = True)

df.drop(122, inplace = True)

df[['day',	'month',	'year',	'Temperature',	'RH',	'Ws']] = df[['day',	'month',	'year',	'Temperature',	'RH',	'Ws']].astype(int)

df[['Rain',	'FFMC',	'DMC',	'DC',	'ISI',	'BUI',	'FWI']] = df[['Rain',	'FFMC',	'DMC',	'DC',	'ISI',	'BUI',	'FWI']].astype(float)

df.drop(['day', 'month', 'year'], axis = 1, inplace = True)

df.head()

df['Classes'] = df['Classes'].str.strip()

df['Classes'].value_counts()

df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0,1)

df.head()

df.info()

x = df.drop('FWI', axis = 1)
y = df['FWI']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25,random_state=42)

x_train.shape

x_train.corr()

def correlation(dataset,thresold):
  corr_col = []
  val = dataset.corr()
  for i in range(len(val.columns)):
    for j in range(i):
      if abs(val.iloc[i,j] > thresold) and (i != j):
        corr_col.append(val.columns[i])

  corr_col  = np.unique(corr_col)
  return corr_col

corr_col = correlation(x_train, 0.85)

print(corr_col)

x_train = x_train.drop(corr_col, axis = 1)
x_test = x_test.drop(corr_col, axis = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred1 = lr.predict(x_test)

print(lr.coef_)

print(lr.intercept_)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mae = mean_absolute_error(y_test,y_pred1)
mse = mean_squared_error(y_test,y_pred1)
score = r2_score(y_test,y_pred1)

print("Mean Absolute Error : ", mae)
print("Mean Squared Error : ", mse)
print("R2 Score : ",  score)

plt.scatter(y_test,y_pred1)
plt.xlabel('Actual')
plt.ylabel('Predicted')

from sklearn.linear_model import Lasso
la = Lasso()
la.fit(x_train,y_train)
y_pred2 = la.predict(x_test)

mae = mean_absolute_error(y_test,y_pred2)
mse = mean_squared_error(y_test,y_pred2)
score = r2_score(y_test,y_pred2)

print(mae)
print(score)

plt.scatter(y_pred2,y_test)



from sklearn.linear_model import LassoCV
lac = LassoCV(cv = 5)
lac.fit(x_train,y_train)
y_pred3 = lac.predict(x_test)

mae = mean_absolute_error(y_test,y_pred3)
mse = mean_squared_error(y_test,y_pred3)
score = r2_score(y_test,y_pred3)

print(mae)
print(score)

plt.scatter(y_pred3,y_test)

from sklearn.linear_model import Ridge
ri = Ridge()
ri.fit(x_train,y_train)
y_pred4 = ri.predict(x_test)

mae = mean_absolute_error(y_test,y_pred4)
mse = mean_squared_error(y_test,y_pred4)
score = r2_score(y_test,y_pred4)

print(mae)
print(score)

plt.scatter(y_pred4,y_test)

from sklearn.linear_model import RidgeCV
ric = RidgeCV(cv = 5)
ric.fit(x_train,y_train)
y_pred5 = ric.predict(x_test)

mae = mean_absolute_error(y_test,y_pred5)
mse = mean_squared_error(y_test,y_pred5)
score = r2_score(y_test,y_pred5)

print(mae)
print(score)

plt.scatter(y_pred5,y_test)

from sklearn.linear_model import ElasticNet
el = ElasticNet()
el.fit(x_train,y_train)
y_pred6 = el.predict(x_test)

mae = mean_absolute_error(y_test,y_pred6)
mse = mean_squared_error(y_test,y_pred6)
score = r2_score(y_test,y_pred6)

print(mae)
print(score)

plt.scatter(y_pred6,y_test)

from sklearn.linear_model import ElasticNetCV
elc = ElasticNetCV()
elc.fit(x_train,y_train)
y_pred7 = elc.predict(x_test)

mae = mean_absolute_error(y_test,y_pred7)
mse = mean_squared_error(y_test,y_pred7)
score = r2_score(y_test,y_pred7)

print(mae)
print(score)

plt.scatter(y_pred7,y_test)