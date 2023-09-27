import sympy as sym
import numpy as np


x = sym.Symbol('x',real=True)



polinomios_lag=[]
raices_final=[]

for n in range(20):
    if n==0:
        polinomios_lag.append(1)
    else:
        f=(sym.exp(-x)*x**n)
        df = [f.diff(x, i) for i in range(n+1)]
        rodrigues = (sym.exp(x) / sym.factorial(n)) * df[n]
        polinomios_lag.append(sym.expand(rodrigues, x))
        
 

for r in polinomios_lag:
    if r==1:
        print ("No existe raiz para L(0)")
    else:
        raices= sym.solve(r, x)
        raices_final.append((np.array((raices))))

  
print(raices_final)
        

  
        
 
