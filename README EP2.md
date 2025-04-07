# Estimação da area de f(x) usando o Método de Monte Carlo.

Este repositório contém um código em Python que utiliza o varias técnicas de Monte Carlo para estimar a area da função f(x). O número de simulações necessárias é determinado com base em número de erro de máximo 4 decimais.
#import numpy as np 

#import matplotlib.pyplot as plt #Usada para plotar gráficos

#import random 

#from scipy.stats import beta, gamma, weibull_min

#from numpy import exp, cos #Facilita a chamada


### def f(x): #Função descrita
    return np.exp(-0.217379 * x) * np.cos(0.24287931874 * x) 

a, b = 0, 1 #Intervalo usado

x_vals = np.linspace(0, 1, 1000) #Gera o gráfico da função, criando 1000 pontos entre 0 e 1

y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label='f(x)')

plt.title('exp(-0.217379 * x) * np.cos(0.24287931874 * x)')

plt.legend()

plt.grid(True)

plt.show()

### def simples(f, a, b, n):
    x = np.random.uniform(a, b, n) #Gera n pontos usando a distribuição Uniforme
    fx = f(x) #Os pontos gerados são testados na função
    return (b - a) * np.mean(fx) #Calcula a media dos valores aleatorios gerados na função e é multiplicada pelo intervalo para estimar a integral.

def hit_or_miss(f, a, b, n):
    x = np.random.uniform(a, b, n) # Gera n pontos ao longo do intervalo [a, b]
    
    y = np.random.uniform(0, 1, n) # Gera n valores de y entre 0 e 1 (altura máxima da função deve ser ≤ 1)
    
    fx = f(x) #Os pontos gerados são testados na função para avaliar aqueles que estão abaixo da curva da f(x)
    
    return (b - a) * 1 * np.mean(y < fx) #A média de resultados é multiplicada pelo intervalo, gerando assim o valor da integral. 

### def importance_sampling(f, dist, dist_pdf, n):

    x = dist.rvs(size=n) #Gera n amostras segundo a distribuição escolhida, por exemplo beta. 
    
    weights = f(x) / dist_pdf(x) #Calcula os pesos corrigidos para cada amostra.
    
    return np.mean(weights) #A média desses pesos corrigidos é a estimativa da integral.




### def control_variate(f, h, integral_h, a, b, n):
    
    x = np.random.uniform(a, b, n) #Gera n pontos usando a distribuição Uniforme
    
    fx = f(x) #Testa os pontos gerados
    
    hx = h(x) #Testa os pontos gerados na função auxiliar
    
    c = np.cov(fx, hx)[0, 1] / np.var(hx) #Calcula o coeficiente para ajustar a variância da estimativa.
    
    I = np.mean(fx - c * (hx - integral_h)) #Aplica a fórmula do método de variável de controle
    
    return (b - a) * I  #Corrige o valor final pelo tamanho do intervalo, já que estamos trabalhando de [a, b].



beta_dist = beta(a=2.0, b=2.0) #É uma ótima oção pois tem o mesmo dominio e o seu gráfico é parecido com f(x))

weibull_dist = weibull_min(c=1.5, scale=0.5) #Permite gerar amostras aleatórias com pârametros que controlam o formato da curva e estica ou comprime a curva no eixo x

beta_pdf = lambda x: beta_dist.pdf(x)#densidade de probabilidade (PDF)

gamma_pdf = lambda x: gamma_dist.pdf(x)#densidade de probabilidade (PDF)

weibull_pdf = lambda x: weibull_dist.pdf(x)#densidade de probabilidade (PDF)


### def h(x): #Função auxiliar para f(x)


  return 1 - x**2 / 2

### def integral_h(): #Integral da função auxiliar para o método de Controle
    
    return 1 - 1/6  

### n = 500000


simples = simples(f, a, b, n) #Chama a função

hitmiss = hit_or_miss(f, a, b, n) #Chama a função

importance = importance_sampling(f, beta_dist, beta_pdf, n) #Chama a função

control = control_variate(f, h, integral_h(), a, b, n) #Chama a função

print(f"Monte Carlo Crude: {simples}")
print(f"Monte Carlo Hit-or-Miss: {hitmiss}")
print(f"Monte Carlo Importance Sampling (Beta): {importance}")
print(f"Monte Carlo Control Variate: {control}")
