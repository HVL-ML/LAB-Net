from pathlib import Path
from PIL import Image
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import skimage

#### Paths ####


def update_paths(path_str):
    return path_str.replace("../../data", str(DATADIR)+"/data")


### Various ####

def compute_dice(pred, targ):

    inter = (pred*targ).float().sum().item()
    union = (pred+targ).float().sum().item()
    
    return 2. * inter/union if union > 0 else None



# Stitch image patches together

def get_concat(images, savefn, kind='image', optimize=True, scale=False):
    """
    Input: 2D array of images in rows and columns
    """
    
    if kind=='image': 
        dst = Image.new('RGB', (5000, 5000))
    elif kind=='mask':
        dst = Image.new('L', (5000, 5000))
    elif kind=='lidar':
        dst = Image.new('F', (5000, 5000))
    
    m,n = images.shape
    for i in range(n):
        for j in range(m):
            dst.paste(Image.open(images[i][j]), (i*500,j*500))
    
    if scale: 
        dst = np.array(dst)*255
        dst = Image.fromarray(dst)
    if optimize:
        dst.save(savefn, optimize=True, quality=70)
    else:
        dst.save(savefn)
        
        


### Analyze buildings ###

def get_building_percentage(maskfn, plot=False):
    mask = Image.open(maskfn)
    if plot: 
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
    total_area = np.prod(mask.size)
    mask_percentage = (np.sum(mask)/total_area)
    return mask_percentage

def count_buildings(maskfn, plot=False):
    mask = Image.open(maskfn)
    if plot: 
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
    return len(np.unique(skimage.measure.label(np.array(mask))))-1
    
