import numpy as np
import matplotlib.pyplot as plt

def mob(u, v):
    x = (a + b*v/2*np.cos(u/2))*np.cos(u)
    y = (a + b*v/2*np.cos(u/2))*np.sin(u)
    z = v/2 * np.sin(u/2)

    return x, y, z

def d(x, y, z, a, b, c):
    return np.sqrt((x-a)**2 + (y-b)**2 + (z-c)**2)

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(-1, 1, 100)

a = 5
b = 1

u, v = np.meshgrid(u, v)

x = (a + b*v/2*np.cos(u/2))*np.cos(u)
y = (a + b*v/2*np.cos(u/2))*np.sin(u)
z = v/2 * np.sin(u/2)

n=100

u_ = np.linspace(0, 2*np.pi, n)
Xn, Yn, Zn = mob(u_, np.array(n*[-1]))
Xp, Yp, Zp = mob(u_, np.array(n*[+1]))

if True:
    ax = plt.axes(projection='3d')
    ax.scatter(Xn,Yn,Zn)
    ax.scatter(Xp,Yp,Zp)


    print(d(Xn, Yn, Zn, Xp, Yp, Zp))

    ax.plot_surface(x, y, z)
    plt.show()
else: 
    plt.plot(u_, d(Xn, Yn, Zn, Xp, Yp, Zp))
    plt.show()
    

