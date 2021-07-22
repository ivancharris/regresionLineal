import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Entrenamiento del Modelo
x = np.array([3008,8971,7632,3431,4536,9451,7909,4917,9812,4539,8146,8029,3400,5550,
            8774,5589,3256,6450,5795,4943,4546,3475,6334,3236,9328,6169,9435,9090, 4131, 3703])
y = np.array([7521, 22429, 19081, 8579, 11341, 23629, 19774, 12294, 24531, 11349, 20366, 20074,
            8501, 13876, 21936, 13974, 8141, 16126, 14489, 12359, 11366, 8689, 15836, 8091, 23321, 15424, 23589, 22726, 10329, 9259])

data = {"x": x, "y": y}
df = pd.DataFrame(data)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Asignando variables modelo de Regresión Lineal

regressionLineal = LinearRegression()
regressionLineal.fit(x, y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)


#Gráfico

fig, ax = plt.subplots(figsize=(7,5), dpi=100)
ax.set_title('Regresión Lineal')
ax.set_xlabel('Megas Agregados')
ax.set_ylabel('Memoria Inicial')
ax = plt.scatter(x, y, s=50)
ax = plt.plot(x, regressionLineal.predict(x), color='red')
plt.show()

#Coeficiente
print(regressionLineal.coef_)
print(regressionLineal.intercept_)
m= regressionLineal.coef_
b= regressionLineal.intercept_

#Resultado
z = m*7000+b
minMem = "%.0f" % z

print("El minimo de memoria de inicio es= ", minMem)

