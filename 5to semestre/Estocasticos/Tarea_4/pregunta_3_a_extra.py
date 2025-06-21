import matplotlib.pyplot as plt

def periodicidad(m, a, c, seed):
    n = 1000000
    instancias_aleatorias = []
    for intento in range(1,n):
        seed = (a * seed + c) % m
        valor_aleatorio = seed / m
        if valor_aleatorio in instancias_aleatorias:
            return intento
        instancias_aleatorias.append(valor_aleatorio)



intento_1 = periodicidad(10007, 23, 17, 123)
intento_2 = periodicidad(10503, 89, 1, 123)
intento_3 = periodicidad(10021, 114, 37, 123)


print("\n")
print(f"Periodicidad para el set 1 después de {intento_1} instancias.")
print(f"Periodicidad para el set 2 después de {intento_2} instancias.")
print(f"Periodicidad para el set 3 después de {intento_3} instancias.")
print("\n")

def obtener_datos(m, a, c, seed, n=1000000):
    datos = []
    for _ in range(n):
        seed = (a * seed + c) % m
        valor_aleatorio = seed / m
        datos.append(valor_aleatorio)
    return datos

datos_1 = obtener_datos(10007, 23, 17, 123)
datos_2 = obtener_datos(10503, 89, 1, 123)
datos_3 = obtener_datos(10021, 114, 37, 123)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(datos_1, bins=100)
plt.title('Distribución set 1')

plt.subplot(1, 3, 2)
plt.hist(datos_2, bins=100)
plt.title('Distribución set 2')

plt.subplot(1, 3, 3)
plt.hist(datos_3, bins=100)
plt.title('Distribución set 3')

plt.tight_layout()
plt.show()
