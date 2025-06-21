import math
import matplotlib.pyplot as plt
import numpy as np
import time



Z = 5 - (math.exp(-1) + math.exp(-2) + math.exp(-3) + math.exp(-4) + math.exp(-5))

F0 = 0
F1 = (1 - math.exp(-1)) / Z
F2 = F1 + (1 - math.exp(-2)) / Z
F3 = F2 + (1 - math.exp(-3)) / Z
F4 = F3 + (1 - math.exp(-4)) / Z
F5 = F4 + (1 - math.exp(-5)) / Z



lista = [F0, F1, F2, F3, F4, F5]
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

datos = []
datos_u = []


reloj = time.perf_counter()

for iteracion in range(10006):
    seed = (a * seed + c) % m
    u = seed / m

    if u < F1:
        i = 1
    elif u >= F1 and u < F2:
        i = 2
    elif u >= F2 and u < F3:
        i = 3
    elif u >= F3 and u < F4:
        i = 4
    elif u >= F4 and u < F5:
        i = 5

    funcion_inversa = (i-1) - (math.log(1 - Z * (u - lista[i-1])) / i)
    datos.append(funcion_inversa)
    datos_u.append(u)

print(f"Tiempo de ejecución: {time.perf_counter() - reloj:.4f} segundos\n")

plt.hist(datos, bins=200, density=True, alpha=0.6, label='Datos simulados')



# Variables para que agregues tus funciones en cada intervalo
x1 = np.linspace(0, 1, 200)
y1 = np.exp(-1 * x1) / Z

x2 = np.linspace(1, 2, 200)
y2 = 2 * np.exp(-2 * (x2- 1)) / Z

x3 = np.linspace(2, 3, 200)
y3 = 3 * np.exp(-3 * (x3 - 2)) / Z

x4 = np.linspace(3, 4, 200)
y4 = 4 * np.exp(-4 * (x4 - 3)) / Z

x5 = np.linspace(4, 5, 200)
y5 = 5 * np.exp(-5 * (x5 - 4)) / Z


plt.plot(x1, y1, color='red', label='Intervalo [0, 1)')
plt.plot(x2, y2, color='blue', label='Intervalo [1, 2)')
plt.plot(x3, y3, color='green', label='Intervalo [2, 3)')
plt.plot(x4, y4, color='orange', label='Intervalo [3, 4)')
plt.plot(x5, y5, color='purple', label='Intervalo [4, 5)')

plt.title('Comparación simulación y distribución teórica')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.legend()
plt.show()


datos_1 = [x for x in datos if 0 <= x < 1]
datos_2 = [x for x in datos if 1 <= x < 2]
datos_3 = [x for x in datos if 2 <= x < 3]
datos_4 = [x for x in datos if 3 <= x < 4]
datos_5 = [x for x in datos if 4 <= x < 5]




print(f"\nprobabilidad empírica vs teórica de caer en el intervalo [0, 1): {len(datos_1) / len(datos)} vs {F1}")
print(f"probabilidad empírica vs teórica de caer en el intervalo [1, 2): {len(datos_2) / len(datos)} vs {F2 - F1}")
print(f"probabilidad empírica vs teórica de caer en el intervalo [2, 3): {len(datos_3) / len(datos)} vs {F3 - F2}")
print(f"probabilidad empírica vs teórica de caer en el intervalo [3, 4): {len(datos_4) / len(datos)} vs {F4 - F3}")
print(f"probabilidad empírica vs teórica de caer en el intervalo [4, 5): {len(datos_5) / len(datos)} vs {F5 - F4}\n")