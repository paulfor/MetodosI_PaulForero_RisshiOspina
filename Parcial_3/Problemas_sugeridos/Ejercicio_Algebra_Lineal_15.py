import numpy as np

# Definimos el conjunto de generadores

sigma_x = np.array([[0, 1], 
                    [1, 0]])

sigma_y = np.array([[0, -1j], 
                    [1j, 0]])
    
sigma_z = np.array([[1, 0], 
                    [0, -1]])

# Se calculan los comutadores

com_xy = np.dot(sigma_x, sigma_y) - np.dot(sigma_y, sigma_x)
com_yz = np.dot(sigma_y, sigma_z) - np.dot(sigma_z, sigma_y)
com_zx = np.dot(sigma_z, sigma_x) - np.dot(sigma_x, sigma_z)

print("Conmutador [σx, σy]:\n", com_xy)
print("\n")
print("Conmutador [σy, σz]:\n", com_yz)
print("\n")
print("Conmutador [σz, σx]:\n", com_zx)
print("\n")
print("El álgebra de Lie si esta dada por la multiplicacion del conmutador entre un 2 escalar")