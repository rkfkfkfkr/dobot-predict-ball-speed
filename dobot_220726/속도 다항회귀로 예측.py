import numpy as np
import numpy.random as rnd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from numpy import array
import pandas as pd

filename = 'ball_data/ball_data1.xlsx'
data = pd.read_excel(filename, engine='openpyxl')
data = np.array(data)

ball_y = []
vel = []

for i in range(len(data)):

    ball_y.append(float(data[i][2]))
    vel.append(float(data[i][3]))


X = np.reshape(ball_y,(-1,1))
y = np.reshape(vel,(-1,1))

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
X_new=np.linspace(float(ball_y[0]), float(ball_y[-1]), 100).reshape(100, 1)

X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

p_y = 132

py = [[p_y, p_y**2]]
pv = lin_reg.predict(py)

print(pv[0][0])

plt.plot(X, y, ".-")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$Ypose$", fontsize=18)
plt.ylabel("$Vel$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.show()

    
