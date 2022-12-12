from pathlib import Path
from PIL import Image
import numpy as np, pandas as  pd, matplotlib.pyplot as plt


#### Plots ####

def plot_image_and_masks_from_df(imgidx, df, figsize=6, with_segm=True):
    imgfn = str(df.iloc[imgidx]['image'])
    if with_segm:
        maskfn = df.iloc[imgidx]['mask']
    f, ax = plt.subplots(figsize=(figsize,figsize))
    img = Image.open(imgfn)
    ax.imshow(img)
    if with_segm:
        mask = Image.open(maskfn)
        ax.imshow(mask, alpha=0.3)
    imgid = imgfn.split("/")[-1].split(".")[0]
    imgsz = img.size
    ax.set_title(f'{imgid}, {str(imgsz)}')
    ax.set_axis_off()
    return ax


def plot_image_lidar_and_masks_from_df(imgidx, df, figsize=6, with_segm=True):
    imgfn = str(df.iloc[imgidx]['image'])
    lidarfn = str(df.iloc[imgidx]['lidar'])
    if with_segm:
        maskfn = df.iloc[imgidx]['mask']
    f, ax = plt.subplots(1,2, figsize=(figsize,figsize))
    img = Image.open(imgfn)
    lidar = np.array(Image.open(lidarfn))
    ax[0].imshow(img)
    ax[1].imshow(img)
    if with_segm:
        mask = Image.open(maskfn)
        ax[0].imshow(mask, alpha=0.3)
        ax[1].imshow(lidar, alpha=0.3)
    imgid = imgfn.split("/")[-1].split(".")[0]
    imgsz = img.size
    ax[0].set_title(f'{imgid}, {str(imgsz)}')
    ax[0].set_axis_off()
    return ax