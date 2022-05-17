from pkgutil import get_data
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils

from matplotlib import collections  as mc

class AnimatedScatter(object):
   """An animated scatter plot using matplotlib.animations.FuncAnimation."""
   def __init__(self, filename):
      numpoints = 50
      self.numpoints = numpoints
      self.stream = self.data_stream()
      self.ds = utils.get_dataset(filename)
      self.timesteps = np.size(self.ds['time'])
      self.Vneighs = self.ds['Vneighs']
      self.vpos = self.ds.variables['pos'][:]
      self.vpos_x = self.vpos[:][::2]
      self.vpos_y = self.vpos[:][1::2]
      self.cell_pos = self.ds['cellPositions'][:] # 7 rows x 400 columns (200 pairs of x,y)
      self.box_side_len = self.ds.variables['BoxMatrix'][0][0];
      # Setup the figure and axes...
      self.fig, self.ax = plt.subplots()
      # Then setup FuncAnimation.
      self.ani = animation.FuncAnimation(self.fig, self.update, interval=200, 
                                          init_func=self.setup_plot, blit=True)

   def setup_plot(self):
      """Initial drawing of the scatter plot."""
      xy =self.ds['cellPositions'][:] 
      x = xy[0][::2]
      y = xy[0][1::2]
      line = []
      curr = 0
      for k in range(0, 1200, 3):
         for j in range(0, 3):
            v = self.Vneighs[0][k+j]
            cur_p = (self.vpos_x[0][curr], self.vpos_y[0][curr])
            p = (self.vpos_x[0][v], self.vpos_y[0][v])
            point_diff = np.subtract(np.asarray(cur_p), np.asarray(p))
            if np.linalg.norm(point_diff) < self.box_side_len/2:
               line.append([cur_p, p]);
            else:
               if point_diff[0] > self.box_side_len/2:
                  point_diff[0] -= self.box_side_len
               if point_diff[0] < -self.box_side_len/2:
                  point_diff[0] += self.box_side_len
               if point_diff[1] > self.box_side_len/2:
                  point_diff[1] -= self.box_side_len
               if point_diff[1] < -self.box_side_len/2:
                  point_diff[1] += self.box_side_len
               line.append([cur_p, utils.tuple_sub(cur_p, point_diff)])
               line.append([p, utils.tuple_add(p, point_diff)])
         curr += 1
      self.scat = self.ax.scatter(x, y, vmin=0, vmax=1,
                                 cmap="jet", edgecolor="k")
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.scat,

   def data_stream(self):
      xy =self.ds['cellPositions'][:] 
      curr = 0
      i = 1
      while i < self.timesteps:
         x = xy[i][::2]
         y = xy[i][1::2]
         line = []
         for k in range(0, 1200, 3):
            for j in range(0, 3):
               v = self.Vneighs[i][k+j]
               cur_p = (self.vpos_x[i][curr], self.vpos_y[i][curr])
               p = (self.vpos_x[i][v], self.vpos_y[i][v])
               point_diff = np.subtract(np.asarray(cur_p), np.asarray(p))
               if np.linalg.norm(point_diff) < self.box_side_len/2:
                  line.append([cur_p, p]);
               else:
                  if point_diff[0] > self.box_side_len/2:
                     point_diff[0] -= self.box_side_len
                  if point_diff[0] < -self.box_side_len/2:
                     point_diff[0] += self.box_side_len
                  if point_diff[1] > self.box_side_len/2:
                     point_diff[1] -= self.box_side_len
                  if point_diff[1] < -self.box_side_len/2:
                     point_diff[1] += self.box_side_len
                  line.append([cur_p, utils.tuple_sub(cur_p, point_diff)])
                  line.append([p, utils.tuple_add(p, point_diff)])
            curr += 1
         i += 1
         print('gh')
         yield np.c_[x, y, line]

   def update(self, i):
      """Update the scatter plot."""
      data = next(self.stream)
      # Set x and y data... (cell positions)
      self.scat.set_offsets(data[0:2])

      lc = mc.LineCollection(data[2], linewidths=1)
      ax.add_collection(lc)
      # We need to return the updated artist for FuncAnimation to draw..
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.scat,


if __name__ == '__main__':
    a = AnimatedScatter('test_p4.000.nc')
    plt.show()
