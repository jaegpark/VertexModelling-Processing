from itertools import count
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
box_side_len = ds.variables['BoxMatrix'][0][0];
print(ds.variables['BoxMatrix'][6])


fig, ax = plt.subplots()
ax.scatter(cell_pos[num_it][::2], cell_pos[num_it][1::2],  c='tab:blue', alpha=0.3, edgecolors='none' )

# TODO: plot lines connecting each vertex
curr = 0

def tuple_add(tuple1, tuple2):
    return tuple(map(lambda x, y: x + y, tuple1, tuple2))

def tuple_sub(tuple1, tuple2):
    return tuple(map(lambda x, y: x - y, tuple1, tuple2))
'''
Notes: vertex indices start from 0 and go to num-1
'''
for i in range(0, 1200, 3):
    for j in range(0, 3):
        v = Vneighs[num_it][i+j]
        cur_p = (vpos_x[curr], vpos_y[curr])
        p = (vpos_x[v], vpos_y[v])
        line = []
        point_diff = np.subtract(np.asarray(cur_p), np.asarray(p))
        if np.linalg.norm(point_diff) < box_side_len/2:
            line.append([cur_p, p]);
        else:
            if point_diff[0] > box_side_len/2:
                point_diff[0] -= box_side_len
            if point_diff[0] < -box_side_len/2:
                point_diff[0] += box_side_len
            if point_diff[1] > box_side_len/2:
                point_diff[1] -= box_side_len
            if point_diff[1] < -box_side_len/2:
                point_diff[1] += box_side_len
            line.append([cur_p, tuple_sub(cur_p, point_diff)])
            line.append([p, tuple_add(p, point_diff)])
        lc = mc.LineCollection(line, linewidths=1)
        ax.add_collection(lc)
    curr += 1

plt.show()

#print(cell_pos[0])


