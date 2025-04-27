# Estimação da função exp(-0.217379 * x) * cos(0.24287931874 * x) usando o Método de Monte Carlo

Este repositório contém um código em Python que utiliza o método de Monte Carlo para estimar o valor de Pi. O número de simulações necessárias é determinado com base em uma margem de erro e um nível de confiança especificado.

Descrição
O código executa os seguintes passos:
1. **Função 'simples(f, a, b, n):'**: Obtem o resultado da função usando o método de Monte Carlo Simples
2. **Função 'hit_or_miss(f, a, b, n):'**: Obtem o resultado da função usando o método de Monte Carlo Hit or Miss
3. **Função 'importance_sampling(f, dist, dist_pdf, n):'**: Obtem o resultado da função usando o método de Monte Carlo Importance Sampling
4. **Função 'control_variate(f, h, integral_h, a, b, n):'**: Obtem o resultado da função usando o método de Monte Carlo Control Variate
5. Executa as funções e exibe o valor estimado da função.

Requisitos
Para executar este código, é necessário instalar as seguintes bibliotecas:


pip install numpy scipy

Execute o seguinte código em Python:


import numpy as np
import random
from scipy.stats import qmc
from scipy.stats import beta, gamma, weibull_min
from numpy import exp, cos

def f(x):
    return np.exp(-0.217379 * x) * np.cos(0.24287931874 * x)

a, b = 0, 1

def simples(f, a, b, n):
    sampler = qmc.Sobol(d=1, scramble=True)
    x = qmc.scale(sampler.random(n), a, b).flatten()
    fx = f(x)
    return (b - a) * np.mean(fx)

def hit_or_miss(f, a, b, n):
    sampler = qmc.Sobol(d=2, scramble=True)
    samples = sampler.random(n)
    scaled_samples = qmc.scale(samples, [a, 0], [b, 1])
    x = scaled_samples[:, 0] 
    y = scaled_samples[:, 1] 
    fx = f(x)
    return (b - a) * 1 * np.mean(y < fx)

def importance_sampling(f, dist, dist_pdf, n):
    x = dist.rvs(size=n)  
    weights = f(x) / dist_pdf(x)  
    return np.mean(weights)

def control_variate(f, h, integral_h, a, b, n):
    sampler = qmc.Sobol(d=1, scramble=True)
    x = qmc.scale(sampler.random(n), a, b).flatten()
    fx = f(x)
    hx = h(x)
    c = np.cov(fx, hx)[0, 1] / np.var(hx)
    I = np.mean(fx - c * (hx - integral_h))
    return (b - a) * I

beta_dist = beta(a=2.0, b=2.0)
weibull_dist = weibull_min(c=1.5, scale=0.5)
beta_pdf = lambda x: beta_dist.pdf(x)
weibull_pdf = lambda x: weibull_dist.pdf(x)

def h(x):
    return 1 - x**2 / 2

def integral_h():
    return 1 - 1/6

n = 500000

simples_result = simples(f, a, b, n)
hitmiss_result = hit_or_miss(f, a, b, n)
importance_result = importance_sampling(f, beta_dist, beta_pdf, n)
control_result = control_variate(f, h, integral_h(), a, b, n)

print(f"Monte Carlo Quase-Aleatório - Crude: {simples_result}")
print(f"Monte Carlo Quase-Aleatório - Hit-or-Miss: {hitmiss_result}")
print(f"Monte Carlo Importance Sampling : {importance_result}")
print(f"Monte Carlo Quase-Aleatório   - Control Variate: {control_result}")
