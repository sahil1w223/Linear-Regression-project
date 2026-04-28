import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/height-weight.csv')

df.head()

x = df.drop('Height', axis = 1)
y = df['Height']

plt.scatter(x,y)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


plt.scatter(x_train,y_train)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

print("The Slope Of Simple Linear Regression : ", lr.coef_)
print("The Intercept Of Simple Linear Regression : ", lr.intercept_)

plt.scatter(x_train,y_train)
plt.plot(x_train, lr.predict(x_train), color = 'blue')

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print("Mean Absolute Error : ", mae)
print("Mean Squared Error : ", mse)
print("Root Mean Squared Error : ", rmse)
print("R2 score : ", score)

adjusted_r_squared = 1-(1-score)*(len(y_test) - 1)/(len(y_test) - x_test.shape[1] - 1)
print("Adjusted R2 score : ", adjusted_r_squared)

new_data_point = 80
new_data_point = sc.transform([[new_data_point]])
# new_data_point = np.array(new_data_point).reshape(-1,1)
predicted_height = lr.predict(new_data_point)
print("Predicted Height : ", predicted_height)
