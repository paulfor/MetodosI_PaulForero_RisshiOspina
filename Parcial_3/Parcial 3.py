import numpy as np

#PUNTO 23


#a Se define la matriz de coeficientes
A=np.array([[0.2, 0.1, 1, 1, 0],\
             [0.1, 4, -1, 1, -1], 
             [1, -1, 60, 0, -2],
             [1, 1, 0, 8, 4],
             [0, -1, -2, 4, 700]])
    
b=np.array([[1], \
            [2], 
            [3],
            [4],
            [5]])
    
    
x_0=np.array([1,0,0,0,0])

# b y c Implementación del algoritmo del gradiente conjugado

def descensoconjugado (A, b, x_0, lr=1e-2):
    Ax=A*x_0
    r_0=Ax-b
    p_0=(-1)*(r_0)
    k=0
    r=[]
    p=[p_0]
    x=[x_0]
    while np.linalg.norm(r[k], np.inf)>lr:  
        
        alfa_k1 = (-1)*((np.transpose(r[k])*p[k])/(np.transpose(p[k])*A*p[k]))
        x_k1 = x_0+(alfa_k1*p[k])
        r_k1 = A*x[k+1]-b
        Betha_k1 = (np.transpose(r_k1)*A*p_0)/(np.transpose(p_0)*A*p_0)
        p_k1 = ((-1)*r_k1)+Betha_k1*p_0
        k+=1
        p.append(p_k1)
        r.append(r_k1)
        x.append(x_k1)
        print(k)
        
    return x[k]
    
print("La solucion es:", descensoconjugado (A,b,x_0))

#d 
