import numpy as np
import matplotlib.pyplot as plt

def torus(a, b):
    phi1 = np.linspace(0, 2*np.pi, 100)
    phi2 = np.linspace(0, 2*np.pi, 100) 

    phi1, phi2 = np.meshgrid(phi1, phi2)

    m = a - b * np.cos(phi2)
    n = b - a * np.cos(phi2)

    x = m*np.sin(phi1)
    y = m*np.cos(phi1)
    z = b*np.sin(phi2)

    ax = plt.axes(projection="3d")
    plt.sca(ax)

    ax.plot_surface(x, y, z)
    plt.show()    


def klein_from_torus(A1, B1, C1, D1, A2, C2):
    """
    A1 > C1
    B1 > D1
    A1 > B1
    C1 > D1

    C2 > A2
    D2 > B2 = D1
    A2 > B2 = D1
    C2 > D2 = B1

    D2 = B1
    B2 = D1
    A1-C1 > 2A2
          = A2+C2
          
    """
    D2 = B1
    B2 = D1

    phi1 = np.linspace(0, 3*np.pi, 200)
    phi2 = np.linspace(0, 2*np.pi, 200)

    P1, P2 = np.meshgrid(phi1, phi2)

    a1 = np.array([(A1 - C1) * np.cos(p1/4)**2 + C1 if p1 <= 2*np.pi else 0 for p1 in phi1])
    b1 = np.array([(B1 - D1) * np.cos(p1/4)**2 + D1 if p1 <= 2*np.pi else 0 for p1 in phi1])

    m1 = a1 - b1 * np.cos(phi2)
    n1 = b1 - a1 * np.cos(phi2)

    a2 = np.array([(C2 - A2) * np.sin(p1/2)**2 + A2 if p1 >= 2*np.pi else 0 for p1 in phi1])
    b2 = np.array([(D2 - B2) * np.sin(p1/2)**2 + B2 if p1 >= 2*np.pi else 0 for p1 in phi1])

    m2 = a2 + b2 * np.cos(phi2)
    n2 = b2 + a2 * np.cos(phi2)

    def cart(P1, P2):
        x = np.zeros(P1.shape) 
        y = np.zeros(P1.shape)
        z = np.zeros(P1.shape)

        for i in range(P1.shape[0]):
            for j in range(P2.shape[1]):
                p1 = P1[i, j]
                p2 = P2[i, j]
                if p1 <= 2*np.pi:
                    x[i, j] = m1[i] * np.sin(p1) 
                    y[i, j] = m1[i] * np.cos(p1)
                    z[i, j] = b1[i] * np.sin(p2)
                else:
                    x[i, j] = m2[i] * np.sin(p1)
                    y[i, j] = A1 - C1 - m2[i]*np.cos(p1)
                    z[i, j] = b2[i] * np.sin(p2)
        return x, y, z

    ax = plt.axes(projection="3d")
    X, Y, Z = cart(P1, P2)
    ax.plot_surface(X, Y, Z)
    plt.show()



klein_from_torus(A1=3, B1=2, C1=1, D1=0.5, A2=0.6, C2=1.4)




def fig8_klein(r):
    r=4 #np.linspace(2, 10, 100)
    theta = np.linspace(0, np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)

    theta, v = np.meshgrid(theta, v)

    c = r + np.cos(theta/2) * np.sin(v) - np.sin(theta/2) * np.sin(2*v)

    x = c * np.cos(theta)
    y = c * np.sin(theta)
    z = np.sin(theta/2) * np.sin(v) + np.cos(theta/2)*np.sin(2*v)

    print(x.shape, y.shape, z.shape)

    #X, Y, Z = np.meshgrid(x, y, z)

    ax = plt.axes(projection="3d")
    plt.sca(ax)

    ax.plot_surface(x, y, z)
    plt.show()