import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import OrderedDict
import glob
import pprint
import cv2
import os

def get_dataset(fname):
    """
    Function that reads nc data from a file using netCDF4 library
    
    return : netCDF4.Dataset : a dictionary containing nc frame data
    """
    return nc.Dataset(fname);
    
def tuple_add(tuple1, tuple2):
    """
    Function that adds two tuples by the following operation:
    (a, b) + (c, d) = (a+c, b+d)

    return : tuple : the tuple sum
    """
    return tuple(map(lambda x, y: x + y, tuple1, tuple2))

def tuple_sub(tuple1, tuple2):
    
    """
    Function that subtracts two tuples by the following operation:
    (a, b) + (c, d) = (a-c, b-d)

    return : tuple : the tuple difference
    """
    return tuple(map(lambda x, y: x - y, tuple1, tuple2))

def seperate_celltype(cellpos, celltypes): 
    """
    cellpos : 1 x 2n array of cell positions
    celltypes: 1 x n array of celltype
    """
    t1x, t1y, t2x, t2y = [], [], [], []
    for i in range(0, len(cellpos)-1,2):
        if celltypes[i//2] == 0:
            t1x.append(cellpos[i])
            t1y.append(cellpos[i+1])
        elif celltypes[i//2] == 1:
            t2x.append(cellpos[i])
            t2y.append(cellpos[i+1])
    return t1x, t1y, t2x, t2y

def read_files():
    """
    Function that reads an nc file directory where each nc file is a timestamp frame

    return : collections.OrderedDict : a dictionary of extracted netCDF4.Dataset frame
             data (in order of appearance).  
    """
    frames = OrderedDict()
    for file in glob.glob("..\data_ME_test\id1\*.nc"):
        frames[file] = get_dataset(file)
    return frames


def draw_frame(frame_num):
    """
    
    """
    global curr, ax, num_edge, t1x, t1y, t2x, t2y

    ax.scatter(t1x, t1y,  c='tab:blue', alpha=0.3, edgecolors='none')
    ax.scatter(t2x, t2y,  c='tab:red', alpha=0.3, edgecolors='none')
    
    
    for i in range(0, num_edge, 3):
        for j in range(0, 3):
            v = Vneighs[frame_num][i+j]
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

def first_item(guh):
    """
    This function peeks the first item in an ordered list type data structure.

    return : first item in a list, dictionary, or other list-type
    """
    return next(iter(guh.items()))

def convert_img_to_mov(image_dir, video_dir):
    """
    This function reads a directory of images and creates a .avi movie file, treating 
    the sequential order of each image as consecutive frames.

    return : none 
    """
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_dir, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    num_it = 0
    fn = read_files()
    #print(first_item(fn))
    
    for i in range(0, 10):
        frame = fn.popitem(False)
        ds = frame[1]       # fn.popitem(False) returns a (key, value) pair of the first element (FIFO order). [1] grabs the values

        Vneighs = ds.variables['Vneighs'][:] # 7 rows x 1200 columns 
        num_edge = Vneighs.shape[1]
        vpos = ds.variables['pos'][:]
        vpos_x = vpos[num_it][::2]
        vpos_y = vpos[num_it][1::2]
        cell_pos = ds['cellPositions'][:] # 7 rows x 400 columns (200 pairs of x,y)
        box_side_len = ds.variables['BoxMatrix'][0][0];
        curr = 0

        t1x, t1y, t2x, t2y = seperate_celltype(cell_pos[num_it], ds.variables['cellType'][num_it])
        
        fig, ax = plt.subplots()

        draw_frame(0)
        print(frame[0][20:-3])
        #plt.show()
        plt.savefig('../frame_images/{fname}.png'.format(fname = frame[0][20:-3]))

    convert_img_to_mov('../frame_images/', '../videos/test_1.avi')

    

