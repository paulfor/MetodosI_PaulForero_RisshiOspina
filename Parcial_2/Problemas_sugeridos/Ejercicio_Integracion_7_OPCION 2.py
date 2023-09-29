import numpy as np


R=1 #Volumen de la esfera

#Se crea una grilla en donde n: número de cuadrados n+1 puntos en cada eje y n^2 cuadrados pequeños que la conforman

#Teniendo en cuenta que z^2 = 1-x^2-y^2 entonces...

z=0
n=10
M=[]

for i in range(n):
    M.append(np.sqrt((1-(i/n)**2-(i/n)**2)))
    for j in range(n):
        M.append(abs(np.sqrt((1-(j/n)**2-(j/n)**2))))
        if 1-(i/n)**2-(j/n)**2>=z:
           M.append(abs(np.sqrt((1-(i/n)**2-(i/n)**2))))
           
V=0 #volumen de la semiesfera
  
for m in M:
    V+=4*((1/n**2)*1/4*(m))
    

print("El valor númerico aproximado del volumen de la semiesfera será:", V)
        
        

