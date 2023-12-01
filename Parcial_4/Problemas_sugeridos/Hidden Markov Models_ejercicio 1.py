import numpy as np

#1

T = np.array([[0.8, 0.2],\
              [0.2, 0.8]])
    
E = np.array([[0.5, 0.9],\
              [0.5, 0.1]])

#a                                                      
pi= np.array([0.2, 0.8])   #distribucion de probabilidad a priori


secuencia = np.array([0, 1, 1, 1, 0, 1, 0, 1]) #Cara se asocia a 1 y sello a 0

probabilidad= np.zeros((len(secuencia), len(pi)))

probabilidad[0] = pi * E[:, secuencia[0]]


for i in range(1, len(secuencia)):
    for j in range(len(pi)):
        probabilidad[i, j] = E[j, secuencia[i]] * np.sum(probabilidad[i-1] * T[:, j])
        
        
print("\n")
#b 
secuencia_mas_probable = np.argmax(probabilidad, axis=1)
probabilidad_secuencia = np.max(probabilidad, axis=1)

print("La secuencia oculta más probable y su probabilidad asociada son:", secuencia_mas_probable, probabilidad_secuencia)
print("\n")

# c
print("La probabilidad de cada estado observable (o) es:", probabilidad)
print("\n")

# d
print("La suma de todos los estados observables es:", np.sum(probabilidad))
print("\n")

# e
print(" El resultado sí depende de la probabilidad a priori ya que afecta a las condiciones iniciales antes de iterar cualquier dato ")
print("\n")
