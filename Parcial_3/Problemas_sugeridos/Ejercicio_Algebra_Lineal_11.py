import numpy as np

H= np.array([[1,2,-1], 
             [1, 0 , 1],
             [4, -4, 5]]) # construimos el Hamiltoniano

v0= np.array([1, 0, 1]) #Otorgamos un vector inicial

# Implementamos el algoritmo de la potencia inversa

inversa_H = np.linalg.inv(H)

def normalizacion(v0):
    norm_v0 = np.linalg.norm(v0)
    v_normal = v0 / norm_v0
    return norm_v0, v_normal

for i in range(100):
    v0 = np.dot(inversa_H, v0)
    E0, Psi_0 = normalizacion(v0)
    
print("E_0 es: ", int(E0))
print("\n")
print("Psi_0 será: ", Psi_0)
