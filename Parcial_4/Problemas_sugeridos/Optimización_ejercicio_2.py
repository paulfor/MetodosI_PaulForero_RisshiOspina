from scipy.optimize import minimize

P_0 = [0, 0, 0]


def funcion(v):
    x,y,z=v
    return x**2 + y**2 + z**2 - 2*z + 1

def restriccion(v):
    x,y,z=v
    return 2*x - 4*y + 5*z - 2

minimo = minimize(funcion, P_0, constraints={'type': 'eq', 'fun': restriccion})


print(minimo)
