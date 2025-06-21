# Celda 1
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import time
import copy

# Celda 2
K = 1000  # Población total
porcentaje_inicial_infectados  = 0.05  # Porcentaje de la población que está expuesta
k = int(K*porcentaje_inicial_infectados)  # Población inicial de infectados, todos sintomáticos
alpha = 0.1  # Tasa de natalidad
beta_i = 0.4  # Tasa de transmisión de infectados sintomáticos
beta_a = 0.15  # Tasa de transmisión de infectados asintomáticos
sigma = 1/7  # Tasa de incubación (7 días en promedio)
p = 0.8  # Probabilidad de que un expuesto se vuelva sintomático
q = 0.15  # Probabilidad de hospitalización al enfermarse
gamma = 1/7  # Tasa aparición sintomas (7 días)
delta = 1/21  # Tasa de recuperación asintomáticos (21 días)
phi = 1/14  # Tasa de hospitalizacion (14 días)
r = 0.95  # Probabilidad de sobrevivir a la hospitalización
mu = 1/28 # Tasa de aislamiento a los recuperados (28 días)
s = 0.9 # Probabilidad de que se recupere completamente
estados_posibles = ["SUSCEPTIBLES", "EXPUESTOS", "INFECTADOS_ASINTOMATICOS" , "INFECTADOS_SINTOMATICOS", "HOSPITALIZADOS", "MUERTOS", "RECUPERADOS"]
estado_inicial = [K-k, 0, 0, k, 0, 0, 0]  # Estado inicial de la población
t_simulacion = 365*1000  # Tiempo de simulación (días)


parametros = {}
parametros["alpha"] = alpha
parametros["beta_i"] = beta_i
parametros["beta_a"] = beta_a
parametros["sigma"] = sigma
parametros["p"] = p
parametros["q"] = q
parametros["gamma"] = gamma
parametros["delta"] = delta
parametros["phi"] = phi
parametros["r"] = r
parametros["mu"] = mu
parametros["s"] = s
parametros["estados_posibles"] = estados_posibles
parametros["estado_inicial"] = estado_inicial
parametros["t_simulacion"] = t_simulacion
parametros["semilla"] = random.randint(0, 10000)  # Semilla aleatoria para la simulación



# Celda 3
def simular_cmtc(parametros):
    # Inicializar la semilla aleatoria
    np.random.seed(parametros["semilla"])
    random.seed(parametros["semilla"])

    estado_actual = np.array(parametros["estado_inicial"])
    tiempo_transcurrido = 0

  # Población inicial de infectados, todos sintomáticos
    alpha = parametros["alpha"]  # Tasa de natalidad
    beta_i = parametros["beta_i"]  # Tasa de transmisión de infectados sintomáticos
    beta_a = parametros["beta_a"]  # Tasa de transmisión de infectados asintomáticos
    sigma = parametros["sigma"]  # Tasa de incubación (7 días en promedio)
    p = parametros["p"]  # Probabilidad de que un expuesto se vuelva sintomático
    q = parametros["q"]  # Probabilidad de hospitalización al enfermarse
    gamma = parametros["gamma"]  # Tasa aparición sintomas (7 días)
    delta = parametros["delta"]  # Tasa de recuperación asintomáticos (21 días)
    phi = parametros["phi"]  # Tasa de hospitalizacion (14 días)
    r = parametros["r"]  # Probabilidad de sobrevivir a la hospitalización
    mu = parametros["mu"] # Tasa de aislamiento a los recuperados (28 días)
    s = parametros["s"] # Probabilidad de que se recupere completamente
    estados_posibles = ["SUSCEPTIBLES", "EXPUESTOS", "INFECTADOS_ASINTOMATICOS" , "INFECTADOS_SINTOMATICOS", "HOSPITALIZADOS", "MUERTOS", "RECUPERADOS"]
    estado_inicial = parametros["estado_inicial"]  # Estado inicial de la población
    t_simulacion = parametros["t_simulacion"]  # Tiempo de simulación (días)
    
    estado_actual = estado_inicial


    estados_historicos = []
    estados_historicos.append(estado_inicial.copy())

    tiempo_de_cambio = [tiempo_transcurrido]

    poblacición_viva_total = []

    poblacion_viva = np.sum(estado_inicial.copy()) - estado_inicial[-2]
    poblacición_viva_total.append(poblacion_viva.copy())

    while tiempo_transcurrido <= t_simulacion:

        N = np.sum(estado_actual) - estado_actual[-1] - estado_actual[-2]

        lambdaa = (beta_i * estado_actual[3] + beta_a * estado_actual[2]) / N


        P_s1 = alpha
        P_se = lambdaa * estado_actual[0]
        P_ea = (1- p) * sigma * estado_actual[1]
        P_as = delta * estado_actual[2]
        P_ei = p * sigma * estado_actual[1]
        P_is = (1 - q) * gamma * estado_actual[3]
        P_ih = q * gamma * estado_actual[3]
        P_hm = (1 - r) * phi * estado_actual[4]
        P_hr = r * phi * estado_actual[4]
        P_rh = (1 - s) * mu * estado_actual[6]
        P_rs = s * mu * estado_actual[6]


    # estados_posibles = ["SUSCEPTIBLES", "EXPUESTOS", "INFECTADOS_ASINTOMATICOS" , "INFECTADOS_SINTOMATICOS", "HOSPITALIZADOS", "MUERTOS", "RECUPERADOS"]





        parametros_exp = []
        parametros_exp.append(P_s1) #0
        parametros_exp.append(P_se) #1
        parametros_exp.append(P_ea) #2
        parametros_exp.append(P_as) #3
        parametros_exp.append(P_ei) #4
        parametros_exp.append(P_is) #5
        parametros_exp.append(P_ih) #6
        parametros_exp.append(P_hm) #7
        parametros_exp.append(P_hr) #8
        parametros_exp.append(P_rh) #9
        parametros_exp.append(P_rs) #10

        suma = np.sum(parametros_exp)


        probabilidades = []
        for parametro in parametros_exp:
            proba = parametro / suma
            probabilidades.append(proba)


        opciones = list(range(11))

        elemento_elegido = random.choices(opciones, weights=probabilidades, k=1)[0]

        if elemento_elegido == 0:
            estado_actual[0] += 1
        
        elif elemento_elegido == 1:
            estado_actual[0] += -1
            estado_actual[1] += 1

        elif elemento_elegido == 2:
            estado_actual[1] += -1
            estado_actual[2] += 1

        elif elemento_elegido == 3:
            estado_actual[2] += -1
            estado_actual[0] += 1

        elif elemento_elegido == 4:
            estado_actual[1] += -1
            estado_actual[3] += 1
        
        elif elemento_elegido == 5:
            estado_actual[3] += -1
            estado_actual[0] += 1

        elif elemento_elegido == 6:
            estado_actual[3] += -1
            estado_actual[4] += 1

        elif elemento_elegido == 7:
            estado_actual[4] += -1
            estado_actual[5] += 1

        elif elemento_elegido == 8:
            estado_actual[4] += -1
            estado_actual[6] += 1

        elif elemento_elegido == 9:
            estado_actual[6] += -1
            estado_actual[4] += 1

        elif elemento_elegido == 10:
            estado_actual[6] += -1
            estado_actual[0] += 1


        delta_tiempo = np.random.exponential(scale=1/suma)

        tiempo_transcurrido += delta_tiempo

        estados_historicos.append(estado_actual.copy())
        tiempo_de_cambio.append(tiempo_transcurrido)

        poblacion_viva = np.sum(estado_inicial.copy()) - estado_inicial[-2]
        poblacición_viva_total.append(poblacion_viva.copy())

    print( "Vector de estados final:", estados_historicos[-1])
    print( "Susceptibles a 365 dias:", estados_historicos[-1][0])
    print( "Expuestos a 365 dias:", estados_historicos[-1][1])
    print( "I. asintomaticos a 365 dias:", estados_historicos[-1][2])
    print( "I. sintomaticos a 365 dias:", estados_historicos[-1][3])
    print( "Hospitalizados a 365 dias:", estados_historicos[-1][4])
    print( "Muertos a 365 dias:", estados_historicos[-1][5])
    print( "Recuperados a 365 dias:", estados_historicos[-1][6])
    print( "Población total viva a 365 dias:", poblacición_viva_total[-1])


    datos = np.array(estados_historicos)
    tiempo = np.array(tiempo_de_cambio)
    vivos_tot = np.array(poblacición_viva_total)

    plt.figure(figsize=(10, 6))


    plt.plot(tiempo, datos[:, 0], label=f'Susceptibles', linestyle="-")
    plt.plot(tiempo, datos[:, 1], label=f'Expuestos', linestyle="-")
    plt.plot(tiempo, datos[:, 2], label=f'I. asintomaticos', linestyle="-")
    plt.plot(tiempo, datos[:, 3], label=f'I. sintomaticos', linestyle="-")
    plt.plot(tiempo, datos[:, 4], label=f'Hospitalizados', linestyle="-")
    # plt.plot(tiempo, datos[:, 5], label=f'Muertos', linestyle="-")
    plt.plot(tiempo, datos[:, 6], label=f'Recuperados', linestyle="-")
    plt.plot(tiempo, vivos_tot, label=f'Poblacion total', linestyle="--")


    prom_susceptibles = 154.82  # Ejemplo: capacidad máxima de hospitales
    plt.axhline(y=prom_susceptibles, color=(random.random(), random.random(), random.random()), linestyle=':', label='prom_susceptibles')

    prom_expuestos = 105.63
    plt.axhline(y=prom_expuestos, color=(random.random(), random.random(), random.random()), linestyle=':', label='prom_expuestos')

    prom_infectados_asintomaticos = 64.06
    plt.axhline(y=prom_infectados_asintomaticos, color=(random.random(), random.random(), random.random()), linestyle=':', label='prom_infectados_asintomaticos')

    prom_infectados_sintomaticos = 84.59
    plt.axhline(y=prom_infectados_sintomaticos, color=(random.random(), random.random(), random.random()), linestyle=':', label='prom_infectados_sintomaticos')

    prom_hospitalizados = 27.27
    plt.axhline(y=prom_hospitalizados, color=(random.random(), random.random(), random.random()), linestyle=':', label='prom_hospitalizados')

    prom_recuperados = 52.92
    plt.axhline(y=prom_recuperados, color=(random.random(), random.random(), random.random()), linestyle=':', label='prom_recuperados')

    # estados_posibles = ["SUSCEPTIBLES", "EXPUESTOS", "INFECTADOS_ASINTOMATICOS" , "INFECTADOS_SINTOMATICOS", "HOSPITALIZADOS", "MUERTOS", "RECUPERADOS"]


    plt.xlabel('Tiempo')
    plt.ylabel('Número de personas')
    plt.title('Evolución de la población en el tiempo')
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()

numero_simulaciones = 1
lista_semillas = [i for i in range(numero_simulaciones)]
tiempo_t =time.time()
for i in lista_semillas:
    parametros["semilla"] = i
    simular_cmtc(parametros)

delta_tiempo = tiempo_t - time.time()
print(delta_tiempo)