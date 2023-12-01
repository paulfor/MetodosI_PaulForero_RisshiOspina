import sympy as sym
from sympy import integrate

x= sym.Symbol("x", real=True)
y= sym.Symbol("y", real=True)
funcion= (2/3)*(x+2*y)

#1
#a
# Para verificar que la función de densidad conjunta sea válida necesitamos que f(x,y) sea mayor o igual a 0 para todo x, y real
# Tambien que las integrales en todo el conjunto dado deben ser igual a 1
 
verificacion=integrate(integrate(funcion, (x, 0, 1)), (y, 0, 1))
print("Si remplazamos un valor de 1 tanto para x como para y obtendremos qué:", (2/3)*((1)+2*(1)))
print("\n")
print("También  si verificamos que las integrales en todo el conjunto den igual a 1 obtendremos como resultado:", verificacion)
print("\n")


#b las distribuciones marginales se hallaran integrando las otras distribuciones unitarias del número de observaciones para cada valor de una de las variables x y y.

g_x = integrate(funcion, (y, 0, 1))
h_y = integrate(funcion, (x, 0, 1))

print("La distribución marginal g(x) será:", g_x.evalf())
print("\n")
print("La distribución marginal h(y) será:", h_y.evalf())
    
#c E(X)=10/8 se debe tener en cuenta que E(X)= integral en todo el conjunto * x * f(x) del cual se utilizarán las marginales g(x) y h(x) para discriminar las variables

E_x = integrate (x*g_x, (x, 0, 1))
print("E(x)=", E_x)
print("\n")

#d E(Y)=11/8
E_y = integrate (y*h_y, (y, 0, 1))
print("E(y)=", E_y)
print("\n")
#e
sigma_xy = integrate(integrate(x*y * funcion, (x, 0, 1)), (y, 0, 1)) - (E_x*E_y)
print("La primera alternativa para calcular la covarianza dará:", (sigma_xy))
print("\n")
#f µ = E(X)
sigma_xy_2= integrate(integrate((x - E_x) * (y - E_y) * funcion, (x, 0, 1)), (y, 0, 1))
print("La segunda alternativa para calcular la covarianza dará:", (sigma_xy_2))
print("\n")
#g
print("Las variables x y y no son independientes dado que la covarianza sigma(X,Y) de cada una de ellas no es nula,es decir, no es igual a 0.")


