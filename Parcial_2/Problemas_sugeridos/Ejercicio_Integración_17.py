import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.Symbol('x',real=True)

def GetLaguerre(n,x):
    if n==0:
        poly = sym.Pow(1,1)
    elif n==1:
        poly = 1-x
    else:
        poly =(((2*n-1-x) * GetLaguerre(n-1, x)) - ((n-1) * GetLaguerre(n-2,x)))/n 
   
    return sym.expand(poly,x)
def GetDLaguerre(n,x):
    Pn = GetLaguerre(n,x)
    return sym.diff(Pn,x,1)

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        try:        
            xn1 = xn - f(xn)/df(xn)
            error = np.abs(f(xn)/df(xn))
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn

def GetRoots(f,df,x,tolerancia = 10):

    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)

        if  type(root)!=bool:
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

def GetAllRootsGLag(n):
    xn = np.linspace(0,n+(n-1)*np.sqrt(n),1000)
   
    Lag = []
    DLag = []
   
    for i in range(n+1):
       Lag.append(GetLaguerre(i,x))
       DLag.append(GetDLaguerre(i,x))
   
    poly = sym.lambdify([x],Lag[n],'numpy')
    Dpoly = sym.lambdify([x],DLag[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
       ValueError('El número de raíces debe ser igual al n del polinomio.')
   
    return Roots
   

def GetWeightsGLag(n):
   
    Roots = GetAllRootsGLag(n)
    W_Laguerre = []
    if n <= len(Roots):
        for k in range(n):
            xk = Roots[k]
            Ln = GetLaguerre(n + 1, xk)
            weight = xk / ((n + 1) ** 2 * Ln ** 2)
            W_Laguerre.append(weight)
    else:
        print("El n dado es mayor que el número de raices")
        return None
    
    Weights=np.array(W_Laguerre)
    Weights_f=np.unique(Weights)
    Weights_final=Weights_f.astype(float)

    return Weights_final


#a 
n=3

funcion = lambda x: (x**3) / (1 - np.exp(-x))

I = 0

for i in range(n):
    raices = GetAllRootsGLag(n)
    pesos = GetWeightsGLag(n)
    I+=pesos[-i-1]*funcion(raices[i])


print("El valor de la integral dará aproximadamente:", I)
    

#b
N=6
Ie = 0
errores=[]
Iteo=(np.pi**4)/15
t=list(range(N))
for k in range(N):
    raices = GetAllRootsGLag(N)
    pesos = GetWeightsGLag(N)
    Ie+=pesos[-k-1]*funcion(raices[k])
    errores.append(Ie/Iteo)
    
plt.figure(figsize=(12, 7))
plt.scatter(t, errores, color="r")
plt.title("Error Porcentual Integral vs. Número de Muestras N")
plt.xlabel("Número de Muestras N")
plt.ylabel("Error Porcentual (%)")
plt.grid(True)
plt.show()