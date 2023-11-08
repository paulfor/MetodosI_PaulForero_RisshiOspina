import numpy as np
from IPython.display import clear_output
import time


G= (lambda x1,x2: np.log(x1**2+x2**2)-np.sin(x1*x2)-np.log(2)-np.log(np.pi), \
    lambda x1,x2: np.exp(x1-x2)+np.cos(x1*x2))
    
G1= ((lambda x1,x2,x3 : (6*x1)-((2*np.cos(x2*x3))-1)), \
     lambda x1,x2,x3 : 9*x2+(np.sqrt((x1**2+np.sin(x3)+1.06)))+0.9, \
     lambda x1,x2,x3: 60*x3+(3*np.exp(-x1*x2))+(10*np.pi)-3)



x0_1=np.array([2,2])
x0_2=np.array([0,0,0])

# Primero Usaremos el método generalizado de Newton Rhapson para el primer sistema
def GetF(G,x):
    
    n = x.shape[0]
    v = np.zeros_like(x)
    
    for i in range(n):
        v[i] = G[i](x[0],x[1])
        
    return v

def Jacobiano(F ,r, e=1e-10):
    N = r.shape[0]
    J = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):  
            rf = r.copy()
            rb = r.copy()           
            rf[j] += e
            rb[j] -= e            
            J[i,j] = (F[i](rf[0],rf[1]) - F[i](rb[0],rb[1]))/(2*e)
            
    return J

def GetNewtonR (G,r,imax=1500, e=1e-10):  
    metric = 1
    v = []
    i = 0
    while metric > e and i < imax:       
        rn = r       
        F = GetF(G,rn)
        J = Jacobiano(G,rn)
        J_inv = np.linalg.inv(J)     
        r = rn - np.dot(J_inv,F)
        dn= r - rn     
        metric = np.max(np.abs(dn))
        v.append(metric) 
        i+= 1
        
    return r,v

#Después utilizamos el método del descenso de gradiente

def Metric(G,r):
    return 0.5 * (np.linalg.norm(GetF(G,r))**2)


def Minimizar(G,r, lr=1e-2 , epochs=int(1e4) , error=1e-7):
    metric = 1
    i = 0
    M = np.array([])
    R = np.array([r])
    
    while metric > error and i < epochs:
        M = np.append(M, Metric(G, r))
        F = GetF(G, r)
        J = Jacobiano(G, r)
        V = GetF(G,r)
        r -= lr * np.dot(J, V)
        R = np.vstack((R, r))
        metric = Metric(G, r)
        i += 1

        if i % 50 == 0:
            clear_output(wait=True)
            time.sleep(0.001)

    return r

r,v = GetNewtonR(G,x0_1)
sol_1N = r
sol_1G = Minimizar(G,x0_1)

print("Las soluciones para el sistema 5.67 utilizando el metodo de Newton es: ", sol_1N)
print("\n")
print("Las soluciones para el sistema 5.67 utilizando el descenso de gradiente es: ", sol_1G)



#Para el segundo sistema haremos lo mismo pero ajustando las escalas
def GetF2(G,x):
    
    n = x.shape[0]
    
    v = np.zeros_like(x)
    
    for i in range(n):
        v[i] = G[i](x[0],x[1], x[2])
        
    return v

def Jacobiano2(F ,r, e=1e-10):
    N = r.shape[0]
    J = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):  
            rf = r.copy()
            rb = r.copy()           
            rf[j] += e
            rb[j] -= e            
            J[i,j] = (F[i](rf[0],rf[1], rf[2]) - F[i](rb[0],rb[1], rb[2]))/(2*e)
            
    return J

def GetNewtonR2 (G,r,imax=1500, e=1e-10):  
    metric = 1
    v = []
    i = 0
    while metric > e and i < imax:       
        rn = r       
        F = GetF2(G,rn)
        J = Jacobiano2(G,rn)
        J_inv = np.linalg.inv(J)     
        r = rn - np.dot(J_inv,F)
        dn= r - rn     
        metric = np.max(np.abs(dn))
        v.append(metric)   
        i+= 1
        
    return r,v

#Después utilizamos el método del descenso de gradiente

def Metric2(G,r):
    return 0.5 * (np.linalg.norm(GetF2(G,r))**2)


def Minimizar2(G, r, lr=1e-2, epochs=int(1e4), error=1e-14):
    metric = 1
    i = 0
    M = np.array([])
    R = np.array([r])
    while metric > error and i < epochs:
        M = np.append(M, Metric2(G, r))
        F = GetF2(G, r)
        J = Jacobiano2(G, r)
        V = np.linalg.solve(J, F)
        r -= lr * V
        R = np.vstack((R, r))
        metric = Metric2(G, r)
        i += 1

        if i % 50 == 0:
            clear_output(wait=True)
            time.sleep(0.001)

    return r


r2,v2 = GetNewtonR2(G,x0_1)
sol_2N = r2
sol_2G = Minimizar2(G1,x0_2)

print("Las soluciones para el sistema 5.67 utilizando el metodo de Newton es: ", sol_2N)
print("\n")
print("Las soluciones para el sistema 5.67 utilizando el descenso de gradiente es: ", sol_2G)
