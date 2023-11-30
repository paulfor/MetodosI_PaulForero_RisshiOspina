import numpy as np


N=int(10e5)
e=0
for i in range (N):
    X=np.random.choice([-1,1], 4)
    X_2= np.sum(X)
    if X_2==0:
        e+=1
        
        
print(" la probabilidad de obtener dos caras y dos sellos es: ", e/N)