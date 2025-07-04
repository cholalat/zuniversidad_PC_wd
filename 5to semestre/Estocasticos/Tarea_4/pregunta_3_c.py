import math
import matplotlib.pyplot as plt
import numpy as np
import time



np.random.seed(2123)

seed = 123


m = 10007
a = 23
c = 17


# m = 10503
# a = 89
# c = 1

# m = 10021
# a = 114
# c = 37

datos_aceptados = []

ce = 1/(1 - 2 * math.exp(-2) + math.exp(-4))

print(ce)

reloj = time.perf_counter()

for iteracion in range(10006):
    seed = (a * seed + c) % m
    u = seed / m

    h1 = np.random.uniform(0, 4)

    if h1 <= 2:
        f1 = ce * h1 * math.exp(-h1)
    else:
        f1 = ce * (4 - h1) * math.exp(-h1)

    g1 = 2 * f1
    if u < g1:
        datos_aceptados.append(h1)


x1 = np.linspace(0, 2, 200)
y1 = ce * x1 * np.exp(-x1)

x2 = np.linspace(2, 4, 200)
y2 = ce * (4 - x2) * np.exp(-x2)




print(len(datos_aceptados))
plt.hist(datos_aceptados, bins=50, density=True, alpha=0.6, label='Datos simulados')
plt.plot(x1, y1, color='red', label='f(x) intervalo [0, 2)')
plt.plot(x2, y2, color='blue', label='f(x) intervalo [0, 4)')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.legend()

plt.show()
