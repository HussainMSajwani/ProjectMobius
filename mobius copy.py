import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider, Button


u = np.linspace(0, 2*np.pi, 1000)
v = np.linspace(-1, 1, 1000)

w = 0.5*np.abs(np.sin(u)+1)

u, v = np.meshgrid(u, v)


x = (1+w*v/2 * np.cos(u/2))*np.cos(u)
y = (1+w*v/2 * np.cos(u/2))*np.sin(u)
z = w*v/2 * np.sin(u/2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.plot_surface(x, y, z, cmap=plt.get_cmap("plasma"))


"""
theta = np.linspace(0, 2*np.pi, 1000)
r0 = 0.7
r = np.linspace(r0, 1/r0, 1000)

r, theta = np.meshgrid(r, theta)

alpha = r - 1/r
beta = r**2 + r**(-2)
gamma = (1/3)*(r**3 - r**(-3))

x = -alpha*np.sin(theta) - beta*np.sin(2*theta) - gamma*np.sin(3*theta)
y = -alpha*np.cos(theta) - beta*np.cos(2*theta) - gamma*np.cos(3*theta)
z = -2*alpha*np.sin(theta)
"""
plt.show()
#for angle in range(0, 360):
 #   ax.view_init(30, angle)
  #  plt.draw()
   # plt.pause(.001)