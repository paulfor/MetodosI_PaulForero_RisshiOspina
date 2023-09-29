import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)
n = 2
Roots,Weights = np.polynomial.laguerre.laggauss(n) #se calculan los ceros y pesos con numpy

polinomios_lag=[]
raices_final=[]

for p in range(n+1):
    if n==0:
        polinomios_lag.append(1)
    else:
        f=(sym.exp(-x)*x**n)
        df = [f.diff(x, i) for i in range(n+1)]
        rodrigues = (sym.exp(x) / sym.factorial(n)) * df[n] #Aqui se utiliza la formula de Rodrigues
        polinomios_lag.append(sym.expand(rodrigues, x))

for r in polinomios_lag:
    if r==1:
        print ("No existe raiz para L(0)")
    else:
        raices= sym.solve(r, x)
        raices_final.append((np.array((raices))))


#Respuestas

#a y "b"
print("El polinomio de Laguerre de orden 2 es:", polinomios_lag[n] )
print("Las raices del polinomio de orden 2 serán:", raices_final[n])


#c

w1=0
w2=0
fw= lambda x: polinomios_lag(2)
for w in range(n):
    pesos = Weights[w]
    w1+=pesos*(fw()) #aqui las bases cardinales eran x1 y x2
    
for w in range(n):
    pesos = Weights[w]
    w2+=pesos*(fw())#aqui las bases cardinales eran x1 y x2

print("Las bases cardinales serán:", w1, w2)

#d
funcion = lambda x: x**3

I = 0

for i in range(n):
    raices = Roots[i]
    pesos = Weights[i]
    I+=pesos*funcion(raices)


print("la regla exacta para un polinomio de grado 3 será:", I)