import netCDF4 as nc
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fn = 'test_p4.000.nc'
ds = nc.Dataset(fn)


#print(ds.variables)
cell_pos = ds['cellPositions'][:] # 7 rows x 400 columns (200 pairs of x,y)

steps = 7
nodes = 100
positions = []
solutions = []

for i in range(steps):
   positions.append(np.random.rand(2, nodes))
   solutions.append(np.random.random(nodes))

fig, ax = plt.subplots()
marker_size = 50

def animate(i):
   fig.clear()
   ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(0, 1), ylim=(0, 1))
   ax.set_xlim(0, 1)
   ax.set_ylim(0, 1)
   s = ax.scatter(cell_pos[i][::2], cell_pos[i][1::2], s=marker_size, c=solutions[i], cmap="RdBu_r", marker="o", edgecolor='black')

plt.grid(b=None)
ani = animation.FuncAnimation(fig, animate, interval=100, frames=range(steps))