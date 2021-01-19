
import platform
import gc

from matplotlib import pyplot as plt
import numpy as np



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