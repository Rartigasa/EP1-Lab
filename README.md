# Estimação de Pi usando o Método de Monte Carlo

Este repositório contém um código em Python que utiliza o método de Monte Carlo para estimar o valor de Pi. O número de simulações necessárias é determinado com base em uma margem de erro e um nível de confiança especificado.

Descrição
O código executa os seguintes passos:
1. **Função `estimar_pi(n)`**: Gera pontos aleatórios dentro de um quadrado e conta quantos caem dentro de um círculo de raio 1.
2. **Função `CalcularN(margem_erro, confianca)`**: Calcula o número de iterações necessárias para atingir uma margem de erro e um nível de confiança desejado.
3. Executa ambas as funções e exibe o valor estimado de Pi.

Requisitos
Para executar este código, é necessário instalar as seguintes bibliotecas:


pip install numpy scipy

Execute o seguinte código em Python:


import numpy as np
from scipy.stats import norm

def estimar_pi(n):
    soma = 0
    for i in range(n):
        x = np.random.rand()
        y = np.random.rand()
        if x**2 + y**2 <= 1:
            soma += 1
    pi = (soma / n) * 4
    return pi

def CalcularN(margem_erro=0.0005, confianca=0.95):
    z = norm.ppf((1 + confianca) / 2)
    n = int((z / (2 * margem_erro))**2)
    return n

valor_n = CalcularN()
pi = estimar_pi(valor_n)

print(f"Pi: {pi}")
