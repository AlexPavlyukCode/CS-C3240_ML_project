import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
import math

# Data preparation and visualisation
data = pd.read_csv('Dataset.csv').fillna(0)

data["Cyclists"] = data.sum(axis=1)
data = data.drop(['Auroransilta', 'Eteläesplanadi', 'Huopalahti (asema)', 'Kaisaniemi/Eläintarhanlahti',
                  'Kaivokatu', 'Kulosaaren silta et.', 'Kulosaaren silta po. ', 'Kuusisaarentie',
                  'Käpylä - Pohjoisbaana', 'Lauttasaaren silta eteläpuoli', 'Merikannontie',
                  'Munkkiniemen silta eteläpuoli', 'Munkkiniemi silta pohjoispuoli', 'Heperian puisto/Ooppera',
                  'Pitkäsilta itäpuoli', 'Pitkäsilta länsipuoli', 'Lauttasaaren silta pohjoispuoli',
                  'Ratapihantie', 'Viikintie', 'Baana'], axis=1)
data = data.rename(columns={"Päivämäärä": "Date"})

data = data.groupby(data.index // 24).sum()

# 0 - 89 are 01.01.2014 to 31.03.2014
# 334 - 454 are 01.12.2014 to 31.03.2015
# 699 - 820 are 01.12.2015 to 31.03.2016
# 1065 - 1185 are 01.12.2016 to 31.03.2017
# 1430 - 1550 are 01.12.2017 to 31.03.2018
# 1795 - 1915 are 01.12.2018 to 31.03.2019
# 2160 - 2281 are 01.12.2019 to 31.03.2020
# 2526 - 2646 are 01.12.2020 to 31.03.2021
# 2891 - 2921 are 01.12.2021 to 31.12.2021

idx = list(range(90, 334)) + list(range(455, 699)) + list(range(821, 1065)) + list(
    range(1186, 1430)) + list(range(1551, 1795)) + list(range(1916, 2160)) + list(range(2282, 2626)) + list(
    range(2647, 2891))
data = data.drop(idx).reset_index(drop=True)

test_set = data.tail(152)
X_test = test_set.index.to_numpy().reshape(-1, 1)
y_test = test_set["Cyclists"].to_numpy()
data.drop(data.tail(152).index, inplace=True)

X = data.index.to_numpy().reshape(-1, 1)
y = data["Cyclists"].to_numpy()

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.scatter(X, y)
ax.set_xlabel("# of the day")
ax.set_ylabel("# of cyclists")
ax.set_title("# of cyclists each day")

plt.show()

# Training/validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Polynomial regression
degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
poly_train_errors = []
poly_val_errors = []
poly_test_errors = []

for deg in degrees:
    train_errors = []
    val_errors = []
    test_errors = []

    for train_index, val_index in cv.split(y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        poly = PolynomialFeatures(degree=deg)
        X_train_poly = poly.fit_transform(X_train)

        lin_regr = LinearRegression(fit_intercept=False)
        lin_regr.fit(X_train_poly, y_train)

        y_pred_train = lin_regr.predict(X_train_poly)
        train_error = mean_squared_error(y_train, y_pred_train)
        train_errors.append(train_error)

        X_val_poly = poly.fit_transform(X_val)
        y_pred_val = lin_regr.predict(X_val_poly)
        val_error = mean_squared_error(y_val, y_pred_val)
        val_errors.append(val_error)

        X_test_poly = poly.fit_transform(X_test)
        y_pred_test = lin_regr.predict(X_test_poly)
        test_error = mean_squared_error(y_test, y_pred_test)
        test_errors.append(test_error)

    avg_train_error = math.log(sum(train_errors) / len(train_errors))
    avg_val_error = math.log(sum(val_errors) / len(val_errors))
    avg_test_error = math.log(sum(test_errors) / len(test_errors))
    poly_train_errors.append(avg_train_error)
    poly_val_errors.append(avg_val_error)
    poly_test_errors.append(avg_test_error)

plt.figure(figsize=(13, 30))

plt.plot(degrees, poly_train_errors, label='Train')
plt.plot(degrees, poly_val_errors, label='Validation')
plt.plot(degrees, poly_test_errors, label='Test')
plt.xticks(degrees)
plt.legend(loc='upper left')

plt.xlabel('Degrees')
plt.ylabel('Loss')
plt.title('Train vs validation vs test loss')
plt.show()

# for i, deg in enumerate(degrees):
#     print("Polynomial regression with degree ", deg, " has training error ", math.log(poly_train_errors[i]), "\n",
#           "Polynomial regression with degree ", deg, " has validation error ", math.log(poly_val_errors[i]), "\n",
#           "Polynomial regression with degree ", deg, " has test error ", math.log(poly_test_errors[i]), "\n")

# MLPRegressor
num_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_neurons = 20
mlp_train_errors = []
mlp_val_errors = []
mlp_test_errors = []

for i, num in enumerate(num_layers):
    train_errors = []
    val_errors = []
    test_errors = []

    for train_index, val_index in cv.split(y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        hidden_layer_sizes = tuple([num_neurons] * num)

        mlp_regr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
        mlp_regr.fit(X_train, y_train)

        y_pred_train = mlp_regr.predict(X_train)
        train_error = mean_squared_error(y_train, y_pred_train)
        train_errors.append(train_error)

        y_pred_val = mlp_regr.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred_val)
        val_errors.append(val_error)

        y_pred_test = mlp_regr.predict(X_test)
        test_error = mean_squared_error(y_test, y_pred_test)
        test_errors.append(test_error)

    avg_train_error = math.log(sum(train_errors) / len(train_errors))
    avg_val_error = math.log(sum(val_errors) / len(val_errors))
    avg_test_error = math.log(sum(test_errors) / len(test_errors))
    mlp_train_errors.append(avg_train_error)
    mlp_val_errors.append(avg_val_error)
    mlp_test_errors.append(avg_test_error)

plt.figure(figsize=(10, 30))

plt.plot(num_layers, mlp_train_errors, label='Train')
plt.plot(num_layers, mlp_val_errors, label='Validation')
plt.plot(num_layers, mlp_test_errors, label='Test')
plt.xticks(num_layers)
plt.legend(loc='upper left')

plt.xlabel('Layers')
plt.ylabel('Loss')
plt.title('Train vs validation vs test loss')
plt.show()

# for i, layer in enumerate(num_layers):
#     print("MLPRegressor with ", layer, " layers has train error ", math.log(mlp_train_errors[i]), "\n",
#           "MLPRegressor with ", layer, " layers has validation error ", math.log(mlp_val_errors[i]), "\n",
#           "MLPRegressor with ", layer, " layers has test error ", math.log(mlp_test_errors[i]), "\n\n")
