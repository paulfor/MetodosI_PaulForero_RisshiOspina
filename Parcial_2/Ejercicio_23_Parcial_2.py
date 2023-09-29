import numpy as np
import sympy as sym
import math
#Ejercicio 23 Radiacion UV proveniente del Sol

#Se utilizara Gauss Laguerre dado que la integral esta definida en 0 e infinito

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)


N=20 #Con estos se encontrara la integral numérica

Roots, Weights = np.polynomial.laguerre.laggauss(N)  #encontrar ceros y pesos con numpy


Lag=[]
h=6.626e-34
kB = 1.3806e-23
c=3e8
T=5772
lamb_0=1e-7
lamb_1=4e-7

#Si c=lambda*v los limites de integracion serán...
v0=c/lamb_0
v1=c/lamb_1

#c
a= (h*v0)/(kB*T)   #Valores numéricos de los limites de integración del numerador
b= (h*v1)/(kB*T)


def GetLaguerre(N,x,y):
    y = sym.exp(-x)*x**N #para que otorguemos la forma de la cuadratura np.exp(-x)
    poly = sym.exp(x)*sym.diff(y,x,N)/( math.factorial(N))
    return poly

for i in range(N+1):
    polinomios = GetLaguerre(i,x,y)
    Lag.append(polinomios)
    
# a y b

f_denominador = lambda x: (x**3) / (1 - np.exp(-x))
f_numerador = lambda x: (x**3) / (np.exp(x)-1)

I_denominador = np.sum( f_denominador(Roots) * Weights)
R, W = np.polynomial.legendre.leggauss(N) 
t = 0.5*((b-a)*R + a + b )
I_numerador = abs(0.5*(b-a)*np.sum(W*f_numerador(t)))

    
#d
f=I_numerador/I_denominador


#e
print("El cálculo de la fracción de rayos UV calculado es:", f)
print("Esta diferencia se debe primordialmente a que el método de cuadratura trae consigo un error porcentual que con valores menores a 100 podría no dar el valor exacto o teórico")