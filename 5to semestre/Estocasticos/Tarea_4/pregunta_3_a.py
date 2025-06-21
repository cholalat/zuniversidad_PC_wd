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