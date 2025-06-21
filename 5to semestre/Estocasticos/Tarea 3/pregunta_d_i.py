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
t_simulacion = 365*10  # Tiempo de simulación (días)


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



# Celda 3
def simular_cmtc(parametros):
    # Inicializar la semilla aleatoria
    np.random.seed(parametros["semilla"])
    random.seed(parametros["semilla"])

    estado_actual = np.array(parametros["estado_inicial"])

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
    estado_inicial = parametros["estado_inicial"].copy()  # Estado inicial de la población
    t_simulacion = parametros["t_simulacion"]  # Tiempo de simulación (días)
    


    tiempo_transcurrido = 0
    estado_actual = estado_inicial.copy()

    tiempo_de_cambio = [tiempo_transcurrido]
    estados_historicos = [estado_actual.copy()]


    poblacición_viva_total = []

    poblacion_viva = np.sum(estado_inicial.copy()) - estado_inicial[-2]
    poblacición_viva_total.append(poblacion_viva.copy())

    while tiempo_transcurrido <= t_simulacion:
        S, E, A, I, H, M, R = estado_actual

        N = S + E + I + A + H

        lambdaa = (beta_i * I + beta_a * A) / N


        P_s1 = alpha
        P_se = lambdaa * S
        P_ea = (1- p) * sigma * E
        P_as = delta * A
        P_ei = p * sigma * E
        P_is = (1 - q) * gamma * I
        P_ih = q * gamma * I
        P_hm = (1 - r) * phi * H
        P_hr = r * phi * H
        P_rh = (1 - s) * mu * R
        P_rs = s * mu * R

        parametros_exp = np.array([
            P_s1,  # 0
            P_se,  # 1
            P_ea,  # 2
            P_as,  # 3
            P_ei,  # 4
            P_is,  # 5
            P_ih,  # 6
            P_hm,  # 7
            P_hr,  # 8
            P_rh,  # 9
            P_rs   # 10
        ])




        suma = P_s1 + P_se + P_ea + P_as + P_ei + P_is + P_ih + P_hm + P_hr + P_rh + P_rs

        if suma == 0:
            break

        delta_tiempo = np.random.exponential(1/suma)
        tiempo_transcurrido += delta_tiempo



        elemento_elegido = random.choices(range(11), weights=parametros_exp, k=1)[0]



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




        estados_historicos.append(estado_actual.copy())
        tiempo_de_cambio.append(tiempo_transcurrido)
        poblacion_viva = S + E + A + I + H + R



        poblacición_viva_total.append(poblacion_viva)

    estados_historicos = np.array(estados_historicos)
    tiempo_de_cambio = np.array(tiempo_de_cambio)

    return estados_historicos[-1], poblacición_viva_total[-1]



numero_simulaciones = 100
lista_semillas = [i for i in range(numero_simulaciones)]
tiempo_t =time.time()



# Celda 6
# Alternativa 1: Aumentar a 1.2 veces la tasa de natilidad
parametros_alternativos_1 = parametros.copy()
parametros_alternativos_1["alpha"] = 1.2*parametros["alpha"]
parametros_alternativos_1["t_simulacion"] = 365*10
lista_semillas = [i for i in range(numero_simulaciones)]

# Nota: Son 100 simulaciones, pero el tiempo de simulación es de 10 años (3650 días), manteniendo el resto de los parámetros constantes del inciso anterior.
# Por cada simulación, se debe cambiar la semilla, para que cada simulación sea diferente. Para eso, cambiar parametros["semilla"] = i, con i el número de la simulación.
# (Es el mismo procedimiento que el inciso anterior, pero ahora se cambia la tasa de natalidad a 1.2 veces la original).




Resultados_finales = []
vivos = []

for i in lista_semillas:
    parametros_alternativos_1["semilla"] = i
    simulacion_i_resultados, poblacion_viva_i = simular_cmtc(parametros_alternativos_1)
    Resultados_finales.append(simulacion_i_resultados)
    vivos.append(poblacion_viva_i)


    if (i+1) % 10 == 0:
        print(f"Simulación {i+1}/{numero_simulaciones} completada.")

delta_tiempo = time.time() - tiempo_t

print("Tiempo ejecucion:", delta_tiempo)
print(parametros_alternativos_1["alpha"])




# Convertir a arrays numpy
resultados = np.array(Resultados_finales)  
vivos = np.array(vivos) 

componentes = ['SUSCEPTIBLES', 'EXPUESTOS', 'INFECTADOS_ASINTOMATICOS', 
               'INFECTADOS_SINTOMATICOS', 'HOSPITALIZADOS', 'MUERTOS', 'RECUPERADOS',
               'POBLACION_VIVA']

datos_completos = np.column_stack((resultados, vivos))

medias = np.mean(datos_completos, axis=0)
desviaciones = np.std(datos_completos, axis=0)
n = len(datos_completos)
intervalos_confianza = 1.96 * desviaciones / np.sqrt(n)  # 95% CI

plt.figure(figsize=(20, 10))

for i in range(8): 
    plt.subplot(2, 4, i+1)  
    
    # Histograma
    counts, bins, patches = plt.hist(datos_completos[:, i], bins=20, alpha=0.7, 
                                   color='skyblue', edgecolor='black')
    
    plt.axvline(medias[i], color='red', linestyle='--', 
               label=f'Media: {medias[i]:.1f}')
    plt.axvline(medias[i] - intervalos_confianza[i], color='green', linestyle=':')
    plt.axvline(medias[i] + intervalos_confianza[i], color='green', linestyle=':', 
               label='95% CI')
    
    plt.title(componentes[i])
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.legend()

plt.tight_layout()


print("\nRESUMEN ESTADÍSTICO COMPLETO")
print(f"{'Componente':<25} {'Media':<10} {'Desviación':<12} Intervalo 95%")
for i in range(8):
    lower = medias[i] - intervalos_confianza[i]
    upper = medias[i] + intervalos_confianza[i]
    print(f"{componentes[i]:<25} {medias[i]:<10.2f} {desviaciones[i]:<12.2f} ({lower:.2f}, {upper:.2f})")


plt.show()