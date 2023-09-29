import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)

def GetLegendre(n,x):
    if n==0:
        poly = 1
    elif n==1:
        poly = x
    else:
        poly = ((2*n-1)*x*GetLegendre(n-1,x)-(n-1)*GetLegendre(n-2,x))/n
   
    return sym.expand(poly,x)


def GetDLegendre(n,x):
    Pn = GetLegendre(n,x)
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

def GetAllRootsGLeg(n):

    xn = np.linspace(-1,1,100)
    
    Legendre = []
    DLegendre = []
    
    for i in range(n+1):
        Legendre.append(GetLegendre(i,x))
        DLegendre.append(GetDLegendre(i,x))
    
    poly = sym.lambdify([x],Legendre[n],'numpy')
    Dpoly = sym.lambdify([x],DLegendre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots

def GetWeightsGLeg(n):

    Roots = GetAllRootsGLeg(n)
    DLegendre = []
    
    for i in range(n+1):
        DLegendre.append(GetDLegendre(i,x))
    
    Dpoly = sym.lambdify([x],DLegendre[n],'numpy')
    Weights = 2/((1-Roots**2)*Dpoly(Roots)**2)
    
    return Weights


TD=300 #Temperatura de Deybe
NoV=0.3

#a

#La integral toma la forma

def funcion(T, delta_T, x):
    return ((np.tanh(np.sqrt(x**2 + delta_T**2))) * (TD/2*T)) / (np.sqrt(x**2+(delta_T**2)))


#b, c y d
n=50
puntos=GetAllRootsGLeg(n)
pesos=GetWeightsGLeg(n)
dT=1e-4
T=1
dT=0.1
I=0
Tc=0

while True:
    if np.abs(I-1/(NoV)) < dT: 
        Tc=T
    else:
        for i in range(n):
            I+=pesos[i]*funcion(T, dT,puntos[i])
        T+=dT

        
print("La temperatura critica será:", Tc, "Kelvin")
