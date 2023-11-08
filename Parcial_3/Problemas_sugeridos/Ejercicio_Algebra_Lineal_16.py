import numpy as np

# Construimos las matrices de Dirac, la métrica del espacio de la relatividad especial y la identidad 4x4

gamma_0 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, -1]])

gamma_1 = np.array([[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, -1, 0, 0],
                  [-1, 0, 0, 0]])

gamma_2 = np.array([[0, 0, 0, -1j],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [-1j, 0, 0, 0]])

gamma_3 = np.array([[0, 0, 1, 0],
                  [0, 0, 0, -1],
                  [-1, 0, 0, 0],
                  [0, 1, 0, 0]])

métrica= np.diag([1, -1, -1, -1])
I_4=np.identity(4)

# Planteamos la relación de anticonmutación
anticonmutacion = (np.dot(gamma_0, gamma_0)) + (np.dot(gamma_1, gamma_1)) + (np.dot(gamma_2, gamma_2)) + (np.dot(gamma_3, gamma_3))

print(anticonmutacion)
print("\n")
print(2*métrica*I_4)
print("\n")
print("El álgebra de Clifford sí se encuentra dada por una relación de anticonmutación")        