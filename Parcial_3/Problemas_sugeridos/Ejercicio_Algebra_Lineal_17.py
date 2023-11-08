import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#a
x,y= sym.symbols("x y", real=True)

#b
z = x + sym.I*y

#c
f_z=(z**3)-1

#d
F = [sym.re(f_z),sym.im(f_z)]

#e
J=sym.Matrix([[F[0].diff(x), F[0].diff(y)],
             [F[1].diff(x), F[1].diff(y)]])

#f
Fn = sym.lambdify((x, y), F, 'numpy')
Jn = sym.lambdify((x, y), J, 'numpy')

#g y h
 
def Newton_Raphson(z0, Fn, Jn, t=1e-7, n_iter=1000):
    for i in range(n_iter):
        x, y = z0
        F = Fn(x, y)
        J = Jn(x, y) 
        zn = z0 - np.linalg.solve(J, F)
        error = np.linalg.norm(zn - z0)
        
        if error < t:
            return zn
        z0 = zn
        
    raise ValueError(" El método de Newton-Raphson no converge ")

z0 = np.array([0.5, 0.5])
print("Si el punto semilla es [0.5, 0.5] el método converge a:", Newton_Raphson(z0, Fn, Jn))


#i y j

N = 50
x1 = np.linspace(-1, 1, N)
y1 = np.linspace(-1, 1, N)
Fractal = np.zeros((N, N), dtype=np.int64)
X, Y = np.meshgrid(x1, y1)

z0 = -0.5 + 0.5j  
z1 = -0.5 - 0.5j  #raices
z2 = 1.0 + 0.0j   

for i in range(N):
    for j in range(N):
        x0, y0 = X[i, j], Y[i, j]
        z0 = x0 + 1j * y0

        for k in range(N):
            if abs(z0 - z0) < 1e-7:
                Fractal[i, j] = 20
            elif abs(z0 - z1) < 1e-7:
                Fractal[i, j] = 100
            elif abs(z0 - z2) < 1e-7:
                Fractal[i, j] = 255

            z0 = z0 - f_z / (3 * z0**2)


    
#k

plt.imshow(Fractal, cmap= "coolwarm" ,extent=[-1,1,-1,1])






