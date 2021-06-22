import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import sympy as sp
from sympy.abc import u, v
from sympy.utilities.lambdify import lambdify

N=300

X = (1+v * sp.cos(u/2))*sp.cos(u)
Y = (1+v * sp.cos(u/2))*sp.sin(u)
Z = v * sp.sin(u/2)

sigma_sym = sp.Array([X, Y, Z])

sigma_u = lambdify((u, v), sigma_sym.diff(u), "numpy")
sigma_v = lambdify((u, v), sigma_sym.diff(v), "numpy")

f = 0.25

sigma = lambda u, v : ((1+w*v/2 * np.cos(u/f))*np.cos(u), (1+w*v/2 * np.cos(u/f))*np.sin(u), w*v/2 * np.sin(u/f))

def generate_data(nbr_iterations):
    dims = (3,1)
    start_positions = np.array(np.sigma(0, 0.5))

    # Computing trajectory
    data = [start_positions]
    for iteration in range(nbr_iterations):
        previous_positions = data[-1]
        new_positions = np.array(sigma(iteration/(2*np.pi), 0.5))
        data.append(new_positions)
    return data

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        iteration = int(iteration)
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters

w = 1

data = [np.array([sigma(u, 0.5)]) for u in np.linspace(0, 2*np.pi, N)]

U = np.linspace(0, 2*np.pi, N)
V = np.linspace(-1, 1, N)

xf, yf, zf = sigma(U, V)
Xu, Yu, Zu = sigma_u(U, V)

U, V = np.meshgrid(U, V)
x, y, z = sigma(U, V)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, z, cmap=plt.get_cmap("plasma"), alpha=0.5)
xs, ys, zs = sigma(np.linspace(0, 2*np.pi, N), 0.5)

ax.plot(xs, ys, zs, color="red")
#ax.quiver(xf[:5], yf[:5], zf[:5], 0.5*(xf+Xu)[:5], 0.5*(yf+Yu)[:5], 0.5*(zf+Zu)[:5])

print(x.shape)

scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

# Number of iterations
iterations = len(data)

if True:
    ani = animation.FuncAnimation(
        fig, animate_scatters, 
        frames=N, interval=0.1,
        fargs=(data, scatters))

if False:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    ani.save('3d-scatted-animated.mp4', writer=writer)

plt.show()