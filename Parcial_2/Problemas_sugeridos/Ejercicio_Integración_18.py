import numpy as np
import sympy as sym
import math

x = sym.Symbol('x',real=True)

#a 

def GetHermite(n,x):
    if n==0:
        poly = sym.Pow(1,1)
    elif n==1:
        poly = 2*x
    else:
        poly= 2*x*GetHermite(n-1,x)-2*(n-1)*GetHermite(n-2,x)
   
    return sym.expand(poly,x)


def GetDHermite(n,x):
    Pn = GetHermite(n,x)
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

def GetAllRootsGHer(n):
    xn = np.linspace(-np.sqrt(4*n+1),np.sqrt(4*n+1),1000)
   
    Her = []
    DHer = []
   
    for i in range(n+1):
       Her.append(GetHermite(i,x))
       DHer.append(GetDHermite(i,x))
   
    poly = sym.lambdify([x],Her[n],'numpy')
    Dpoly = sym.lambdify([x],DHer[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
       ValueError('El número de raíces debe ser igual al n del polinomio.')
   
    return Roots
   
   
def GetWeightsGHer(n):
    
    Roots=GetAllRootsGHer(n)
    Hermite=[]
    
    for i in range(n+1):
        Hermite.append(GetHermite(i,x))
    
    Poly=sym.lambdify([x],Hermite[n-1],'numpy')
   
    Weights = ((2**(n-1)) * (np.math.factorial(n) * np.sqrt(np.pi))) / ((n**2)*(Poly(Roots)**2))
    Weights_final= np.unique(Weights)
    
    return Weights_final

print("Los primeros 20 ceros de los polinomios serán:", GetAllRootsGHer(20))
print("Sus pesos correspondientes serán:", GetWeightsGHer(20))

#b

#la raiz del polinomio 1 de Hermite es 0

def psi1 (n, x):
    return (1/np.sqrt(2**n*math.factorial(n)))*(1/np.pi**2)*(sym.exp(-x**2/2)*GetHermite(1,x))


valor_aprox= np.sum(GetWeightsGHer(1)*abs(psi1(1,x))**2*x**2)

print("el valor cuadratico medio de la posicion de la partıcula en el primer estado exitado es:", valor_aprox)
                    



