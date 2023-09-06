import math


#a y b

def metodo_muller (f, x0, x1, x2, itmax):
    it=0
    while it<=itmax:
        try: 
            h0 = x1 - x0
            h1 = x2 - x1
            f0 = f(x0)
            f1 = f(x1)
            f2 = f(x2)
            a = (f1 - f0) / h0
            b = (f2 - f1) / h1
            c = (b - a) / (h1 + h0)
            x3=-2*c/b+math.sqrt(b**2-4*a*c)
        except ZeroDivisionError:
           print("No se puede dividir por cero")
         
    it+=1
    determinante=math.sqrt(b**2-4*a*c)
    if abs(b + determinante) > abs(b - determinante):
        E = b + determinante
    else:
        E = b - determinante
    h = f2 * -2/ E
    t = x2 + h
    
    
#d
#e=f(x2)<1e-10

    
