
import platform
import gc

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math
from math import cos, radians
from PIL import Image



# Extract layer RGB, IR & Mask from image
def extract_layers(img):
    h, w, d = img.shape
    
    # Layer mask
    layer_mask = np.copy(img[:, :, -1]).astype(np.uint8)

    # Layer RGB
    if d == 5 or d == 4:
        # Change RGB value to 0~255, then change precision to save space
        if img[:, :, 0:3].max() > 255:
            layer_RGB = np.copy(img[:, :, 0:3] / 257).astype(np.uint8)
        elif img[:, :, 0:3].max() > 1:
            layer_RGB = np.copy(img[:, :, 0:3]).astype(np.uint8)
        else:
            layer_RGB = np.copy(img[:, :, 0:3] * 255).astype(np.uint8)
        # Set background to black
        layer_RGB[np.where(layer_mask == 0)] = 0

    # Layer IR
    if d == 5:
        layer_IR = np.copy(img[:, :, 3])
        # Set background to black
        layer_IR[np.where(layer_mask == 0)] = 0
    elif d == 2:
        layer_IR = np.copy(img[:, :, 0])
        # Set background to black
        layer_IR[np.where(layer_mask == 0)] = 0
    
    if d == 4:
        layer_IR = []
    elif d == 2:
        layer_RGB = []
        
    return(layer_RGB, layer_IR, layer_mask)


def low_res(img, scale_factor):
    h, w, d = img.shape
    if img != []:
        length = h/scale_factor if h>w else w/scale_factor
        size = [length, length]
        img_low_res = Image.fromarray(img)
        img_low_res.thumbnail(size, Image.ANTIALIAS)
        img_low_res = np.asarray(img_low_res)
    else:
        img_low_res = []
    return img_low_res


def RGB2HSV(layer_RGB):
    layer_HSV = matplotlib.colors.rgb_to_hsv(layer_RGB)
    layer_HSV[:, :, 0] = layer_HSV[:, :, 0] * 360
    layer_HSV[:, :, 2] = layer_HSV[:, :, 2] / 255
    
    return(layer_HSV)


def DGCI(layer_HSV):
    layer_DGCI = (layer_HSV[:, :, 0] / 60 + (1 - layer_HSV[:, :, 1]) + (1 - layer_HSV[:, :, 2])) / 3
    
    return(layer_DGCI)


# Play sound when done
def done():
    print('\nDone')
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(500, 1500)
    elif platform.system() == 'Darwin':
        import os
        os.system( "say Done" )
    elif platform == 'linux' or platform == 'linux2':
        import os
        os.system( "say Done" )
    # Reduce use of memory
    gc.collect()
    

def pix2geo(pix_loc, gt):
    a, b, c, d, e, f = gt
    x, y = pix_loc
    xgeo = a + x*b + y*c
    ygeo = d + x*e + y*f
    return([xgeo, ygeo])
    
    
def geo2pix(geo_loc, gt):
    a, b, c, d, e, f = gt
    xgeo, ygeo = geo_loc
    x = (xgeo - a) / b
    y = (ygeo - d) / f
    return([x, y])


def meter_per_pix(gt):
    lat = radians(gt[3])
    lon_degree_per_pix = gt[1]
    lat_degree_per_pix = -gt[5]

    lon_meter_per_degree = 111412.84 * cos(lat) - 93.5 * cos(3 * lat) + 0.118 * cos(5 * lat)
    lat_meter_per_degree = 111132.92 - 559.82 * cos(2 * lat) + 1.175 * cos(4 * lat) - 0.0023 * cos(6 * lat)
    
    lon_meter_per_pix = lon_meter_per_degree * lon_degree_per_pix
    lat_meter_per_pix = lat_meter_per_degree * lat_degree_per_pix

    return(lon_meter_per_pix, lat_meter_per_pix)


def plotVGPS2plotV(plot_vertices_gps, gt):
    plot_vertices = {}
    for plot_name in plot_vertices_gps.keys():
        one_plot_vertices_gps = plot_vertices_gps[plot_name]
        one_plot_vertices = []
        for vertex in one_plot_vertices_gps:
            pix_loc = geo2pix(vertex, gt)
            one_plot_vertices.append(pix_loc)
        one_plot_vertices = np.array(one_plot_vertices)
        one_plot_vertices = np.round(one_plot_vertices)
        one_plot_vertices = one_plot_vertices.astype(int)
        plot_vertices[plot_name] = (one_plot_vertices)
    return plot_vertices


def hist_of_img(img, bins):
    fig, ax = plt.subplots(figsize=(5, 5))
    flattened_img = img.flatten()
    ax.hist(flattened_img, bins=bins)
    
    
# Shrink color range for plot if possible
def cal_vmin_vmax(img, mask):
    mask_not_0_inds = np.where(mask > 0)
    mean = np.mean(img[mask_not_0_inds])
    std = np.std(img[mask_not_0_inds])
    three_left = mean - 3 * std
    three_right = mean + 3 * std
    allmin = img[mask_not_0_inds].min()
    allmax = img[mask_not_0_inds].max()
    vmin = allmin if allmin > three_left else three_left
    vmax = allmax if allmax < three_right else three_right
    return vmin, vmax


def fig_size(h, w):
    ratio = h/w
    if ratio >= 2:
        fig_size = [6, 10]
    elif ratio > 0.5 and ratio <2:
        fig_size = [9, 9]
    else:
        fig_size = [10, 6]
        
#     return(fig_size)
    return([10, 10])


def find_point_in_img(point_ref, mtp, mtp_ref, shift_geo):    
    diff = mtp_ref - point_ref
    dist = np.sum(np.square(diff), axis=1)
    if 0 in dist:
        shift_coef = np.zeros(len(diff))
        ind = np.where(dist==0)
        shift_coef[ind] = 1
    else:
        dist_inv = 1 / dist
        sum_dist_inv = np.sum(dist_inv)
        shift_coef = dist_inv / sum_dist_inv
    shift_coef = np.expand_dims(shift_coef, 1)
    point_shift_geo= np.sum(np.multiply(shift_geo, shift_coef), 0)
    return(point_shift_geo)
