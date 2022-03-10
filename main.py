import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
print(data.head)

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

X_demo = data.index.to_numpy().reshape(-1, 1)
y_demo = data["Cyclists"].to_numpy()

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.scatter(X_demo, y_demo)
ax.set_xlabel("# of the day")
ax.set_ylabel("# of cyclists")
ax.set_title("# of cyclists each day")

plt.show()

# Polynomial regression
degrees = [2, 3, 4]
tr_errors = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X_demo, y_demo)

    lin_regr = LinearRegression()
    lin_regr.fit(X_poly, y_demo)

    y_pred = lin_regr.predict(X_poly)
    tr_error = mean_squared_error(y_demo, y_pred)
    tr_errors.append(tr_error)

    X_fit = np.linspace(0, 1000, 100)
    plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))),
             label="learnt hypothesis")  # plot the polynomial regression model
    plt.scatter(X_demo, y_demo, color="b", s=10,
                label="Datapoints ")
    plt.xlabel('# of the day')
    plt.ylabel('# of cyclists')
    plt.legend(loc="best")
    plt.title('Training error = {:.5}'.format(tr_error))
    plt.show()
