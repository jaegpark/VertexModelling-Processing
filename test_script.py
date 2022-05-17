from itertools import count
from turtle import numinput
import netCDF4 as nc
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections  as mc


num_it = 0
fn = 'test_p4.000.nc'
ds = nc.Dataset(fn)
Vneighs = ds.variables['Vneighs'][:] # 7 rows x 1200 columns 
vpos = ds.variables['pos'][:]
vpos_x = vpos[num_it][::2]
vpos_y = vpos[num_it][1::2]
cell_pos = ds['cellPositions'][:] # 7 rows x 400 columns (200 pairs of x,y)

print(vpos.shape)


fig, ax = plt.subplots()
ax.scatter(cell_pos[num_it][::2], cell_pos[num_it][1::2],  c='tab:blue', alpha=0.3, edgecolors='none' )

# TODO: plot lines connecting each vertex
curr = 0

'''
Notes: vertex indices start from 0 and go to num-1
'''
for i in range(0, 1200, 3):
    v1 = Vneighs[num_it][i]
    v2 = Vneighs[num_it][i+1]
    v3 = Vneighs[num_it][i+2]
    cur_p = (vpos_x[curr], vpos_y[curr])
    p1 = (vpos_x[v1], vpos_y[v1])
    p2 = (vpos_x[v2], vpos_y[v2])
    p3 = (vpos_x[v3], vpos_y[v3])
    #print(v1)
    lines = [[cur_p, p1], [cur_p, p2], [cur_p, p3]]
    lc = mc.LineCollection(lines, linewidths=1)
    ax.add_collection(lc)
    curr += 1

plt.show()

#print(cell_pos[0])


