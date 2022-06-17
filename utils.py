from tkinter import Toplevel
from turtle import right
from types import CellType
from cv2 import rotate
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from collections import OrderedDict
import glob
import cv2
import os
from PIL import Image

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
    global curr, ax, num_edge, t1x, t1y, t2x, t2y, num_cell, mesectoderm_vertices

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
            
            # check if mesectoderm cell edge           
            if curr in mesectoderm_vertices and v in mesectoderm_vertices:
                lc = mc.LineCollection(line, linewidths=1, colors=colors.to_rgba('Crimson'))
            else:
                lc = mc.LineCollection(line, linewidths=1, colors=(0, 0, 0, 1))
            
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

def get_mesectoderm_cell_indices(numcell, celltypelist):
    """
    
    return : list of cell indices of the mesectoderm
    """
    ans = []
    for i in range(numcell):
        if celltypelist[i] == 1:
            ans.append(i)
    return ans

def get_mesectoderm_vertex_coords(mes_vert_idx, vert_px, vert_py, ax):
    x, y = [], []
    for i in range(0, len(mes_vert_idx)):
        x.append(vert_px[mes_vert_idx[i]])
        y.append(vert_py[mes_vert_idx[i]])
    return x, y
    

def get_mesectoderm_vertex_indices(num_v, mes_celllist, Vcellneigh):
    vert_idx = []
    for i in range(0, num_v):
        if (Vcellneigh[3 * i] in mes_celllist) or (Vcellneigh[3*i + 1] in mes_celllist) or (Vcellneigh[3*i+2] in mes_celllist):
            vert_idx.append(i)
    return vert_idx

def get_cell_vertices(cell_num, cell_vertices):
    ans = []
    for i in range(len(cell_vertices[0])):
        if cell_vertices[cell_num][i] != -1:
            ans.append(cell_vertices[cell_num][i])
    ans = [int(a) for a in ans]
    return ans

def get_vertex_coords(vertex_ind, vposx, vposy):
    if isinstance(vertex_ind, list):
        xs, ys = [], []
        for i in range(len(vertex_ind)):
            xs.append(vposx[vertex_ind[i]])
            ys.append(vposy[vertex_ind[i]])
        return xs, ys
    else:
        return vposx[vertex_ind], vposy[vertex_ind]
    
def draw_mesectoderm_filled(mesectoderm_cell_indices, cell_vertices, vposx, vposy, ax):
    patches = []
    if isinstance(mesectoderm_cell_indices, list):
        for idx in mesectoderm_cell_indices:
            vertices = get_cell_vertices(idx, cell_vertices)
            coordsx, coordsy = get_vertex_coords(vertices, vposx, vposy)
            poly = Polygon(np.c_[coordsx, coordsy], closed=True)
            patches.append(poly)
            #ax.fill(coordsx, coordsy, facecolor='lightsalmon')
    else:
        vertices = get_cell_vertices(mesectoderm_cell_indices, cell_vertices)
        coordsx, coordsy = get_vertex_coords(vertices, vposx, vposy)
        poly = Polygon(np.c_[coordsx, coordsy], closed=True)
        patches.append(poly)
        #ax.fill(coordsx, coordsy, facecolor='lightsalmon')
    p = PatchCollection(patches, alpha=0.4)
    ax.add_collection(p)

def find_mesectoderm_boundary(numv, Vneighs, Vcellneighs, cellType, vposx, vposy):
    """
    get list of vertices along boundary edge
    """
    Vneighs_trans = np.reshape(Vneighs, (-1, 3))
    Vcellneighs_trans = np.reshape(Vcellneighs, (-1, 3))
    mes_ver_list = []
    mes_lines = []
    for i in range(numv):
        curr_v_neighbours = Vneighs_trans[i]
        curr_v_cellneighs = Vcellneighs_trans[i]
        for neighbour in curr_v_neighbours:
            second_v_cellneighs = Vcellneighs_trans[neighbour]
            common_cell_neighbours = list(set(curr_v_cellneighs).intersection(second_v_cellneighs))
            #print(common_cell_neighbours)
            temp_mes_count, temp_non_count = 0, 0
            # Check if it has 1 mesectoderm
            '''
            check if both cells have same mesectoderm neighbour
            check if both cells have at least 1 mesectoderm and 1 nonmesectoderm neighbour
            '''

            for cell in common_cell_neighbours:
                if cellType[cell] == 1:
                    temp_mes_count += 1
                else:
                    temp_non_count += 1
            
            if temp_mes_count == 1 and temp_non_count == 1:

                mes_ver_list.append(i)
                mes_ver_list.append(neighbour)
                mes_lines.append([(vposx[i], vposy[i]), (vposx[neighbour], vposy[neighbour])])
                # TODO: improve efficiency by removing double counting of lines
    return set(mes_ver_list), mes_lines
            
def draw_line(p1, p2, colour, ax):
    
    """
    draws a line given two coordinate tuples, following the periodic boundary conditions
    """
    global box_side_len
    line = []
    point_diff = np.subtract(np.asarray(p1), np.asarray(p2))
    if np.linalg.norm(point_diff) < box_side_len/2:
        line.append([p1, p2]);
    else:
        if point_diff[0] > box_side_len/2:
            point_diff[0] -= box_side_len
        if point_diff[0] < -box_side_len/2:
            point_diff[0] += box_side_len
        if point_diff[1] > box_side_len/2:
            point_diff[1] -= box_side_len
        if point_diff[1] < -box_side_len/2:
            point_diff[1] += box_side_len
        line.append([p1, tuple_sub(p1, point_diff)])
        line.append([p2, tuple_add(p2, point_diff)])
    
    lc = mc.LineCollection(line, linewidths=1, colors=colors.to_rgba(colour))
    ax.add_collection(lc)


def draw_mesectoderm_vertices(vposx, vposy, mes_vertices, ax):
    x = [vposx[a] for a in mes_vertices]
    y = [vposy[a] for a in mes_vertices]
    ax.scatter(x, y, c="tab:green", s=0.1)
    return x, y

def sweeper(img_path):
    """
    sweeper takes in an image and returns groupings of pixels corresponding to the top/bottom boundary of the mesectoderm
    """
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th, im_gray_th = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)
    height, width = im_gray_th.shape
    upper_boundary = []
    lower_boundary = []

    for i in range(width):
        column = []
        for j in range(height):
            # found pixel
            #print(im_gray_th[j])
            if im_gray_th[j][i] == 0:   # TODO: change this to actual y value
                column.append([i, j])

        # post process column by taking the first half as upper mes, second half as lower mes  
        # THIS METHOD DOES NOT WORK... WILL INSTEAD TAKE TOPMOST POINT AND BOTTOMMOST POINT      
        #upper = np.average(column[:len(column)//2], axis=1)
        #lower = np.average(column[len(column)//2:], axis=1)
        upper = column[0]
        lower = column[-1]
        upper_boundary.append(upper)  # list of points, i'th list contains all points corresponding to the i'th column in the image
        lower_boundary.append(lower)

    cv2.imwrite('otsu.jpg', im_gray_th)
    return upper_boundary, lower_boundary
    
def step_func(val):
    return 1 if val <= 3 else 0

def calc_pixel_widths(upper_tuple_list, lower_tuple_list, N, width):
    distances = []
    for i in range(width):
        mean_top = np.mean(upper_tuple_list[i])
        mean_bot = np.mean(lower_tuple_list[i])
        distances.append(mean_top-mean_bot)
    
    return distances

def calc_roughness(rotated_points):
    numpixels = len(rotated_points)
    '''
    First, find h bar
    '''
    # h-bar calculations:
    xaxis = rotate_points[0][1]
    h_tot = 0
    for i in range(len(rotated_points)):
        h_tot += np.abs(rotated_points[i][1] - xaxis)
    h_bar = h_tot / numpixels

    '''
    Now find w^2
    '''
    temp = 0
    for i in range(len(rotated_points)):
        h_i = np.abs(rotated_points[i][1] - xaxis)
        temp += np.power(h_i-h_bar, 2)
    w2 = temp/numpixels

    return np.sqrt(w2)
    
def calc_mes_internalization(upper_list, lower_list):
    distances = np.delete(np.subtract(upper_list, lower_list), 0, 1)
    s = [step_func(x) for x in distances]
    ans = np.sum(s)/len(upper_list)
    return ans

def rotate_points(upper_points_list, bottom_points_list, N):
    top_left = upper_points_list[0]
    top_right = upper_points_list[-1]
    bot_left = bottom_points_list[0]
    bot_right = bottom_points_list[-1]
   
    #print(top_left[0], top_left[1], top_right[0], top_right[1])
    slope_top = (top_right[1] - top_left[1])/(top_right[0] - top_left[0])
    print(slope_top)
    slope_bot = (bot_right[1] - bot_left[1])/(bot_right[0] - bot_left[0])
    theta_r_top = np.arctan(slope_top)
    theta_r_bot = np.arctan(slope_bot)
    if slope_top < 0:
        theta_r_top = - theta_r_top
    if slope_bot < 0:
        theta_r_bot = - theta_r_bot
    Rt = [[np.cos(theta_r_top), -np.sin(theta_r_top)], 
         [np.sin(theta_r_top), np.cos(theta_r_top)]]
    Rb = [[np.cos(theta_r_bot), -np.sin(theta_r_bot)], 
         [np.sin(theta_r_bot), np.cos(theta_r_bot)]]
    
    for i in range(len(upper_points_list)):
        upper_points_list[i] = np.matmul(Rt, upper_points_list[i])
    
    for i in range(len(bottom_points_list)):
        bottom_points_list[i] = np.matmul(Rb, bottom_points_list[i])

    return upper_points_list, bottom_points_list

if __name__ == "__main__":
    num_it = 0
    fn = read_files()
    
    for i in range(0, 1):
        frame = fn.popitem(False)
        ds = frame[1]       # fn.popitem(False) returns a (key, value) pair of the first element (FIFO order). [1] grabs the values
        num_cell = ds.dimensions['Nc'].size
        num_v = ds.dimensions['Nv'].size
        
        VCellNeighs = ds.variables['VertexCellNeighbors'][:][0] # size of 3 * num_v : groups of 3 (3 cell neighbours for each vertex)
        cellVertices = np.reshape(ds.variables['cellVer'][:][0], (-1, 16))  # 2D array: i-th row has the list of vertex indices of the i-th cell 
        cellVerNum = ds.variables['cellVerNum'][:]
        Vneighs = ds.variables['Vneighs'][:] 
        num_edge = Vneighs.shape[1]
        vpos = ds.variables['pos'][:]
        vpos_x = vpos[num_it][::2]
        vpos_y = vpos[num_it][1::2]
        cell_pos = ds['cellPositions'][:] 
        cell_types = ds['cellType'][num_it]
        box_side_len = ds.variables['BoxMatrix'][0][0];
        curr = 0
        fig, ax = plt.subplots()

        #print(len(cell_types))
        #print(num_cell)
        ''' Get Processed Data '''
        mesectoderm_cells = get_mesectoderm_cell_indices(numcell=num_cell, celltypelist=cell_types)
        mesectoderm_vertices = get_mesectoderm_vertex_indices(num_v, mesectoderm_cells, VCellNeighs) 
        mesectoderm_boundary_vertices, mesectoderm_boundary_lines = find_mesectoderm_boundary(num_v, Vneighs, VCellNeighs, cell_types, vpos_x, vpos_y)
        mesectoderm_vertex_coords_x , mesectoderm_vertex_coords_y = get_mesectoderm_vertex_coords(mesectoderm_vertices, vpos_x, vpos_y, ax)


        t1x, t1y, t2x, t2y = seperate_celltype(cell_pos[num_it], ds.variables['cellType'][num_it])
        

        #draw_frame(0)
     
        bcoords_x, bcoords_y = draw_mesectoderm_vertices(vpos_x, vpos_y, mesectoderm_boundary_vertices, ax)    # ALSO DRAWS THE BOUNDARY
        plt.axis([0, 20, 8, 12])
        for lines in mesectoderm_boundary_lines:
            draw_line(lines[0], lines[1], 'tab:green', ax)
        
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #plt.show()
        filename = '../frame_images/{fname}.png'.format(fname = frame[0][20:-3])
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    #convert_img_to_mov('../frame_images/', '../videos/test_1.avi')

    '''
        TODO: 
        3. find a way to convert the saved images and the coordinates...
    '''
    upperboundary, lowerboundary = sweeper(filename)
    r_upper_boundary, r_lower_boundary = rotate_points(upper_points_list=upperboundary, bottom_points_list=lowerboundary, N=1)
    