import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

cluster = True

if cluster:
    sys.path.insert(0, '/auto/home/users/m/d/mdausort/Software/large_image_source_tifffile')
else:
    sys.path.insert(0, 'C:/Users/dausort/OneDrive - UCL/Bureau/large_image_source_tifffile')
    
import large_image

filename = "/CECI/home/users/m/d/mdausort/Cytology/Training/15C00282.ndpi"

count_cond = 0
count_tot = 0

for slide_info in tqdm(large_image.getTileSource(filename).tileIterator(
    scale=dict(magnification=20),
    tile_size=dict(width=224, height=224),
    tile_overlap=dict(x=0, y=0),
        format=large_image.tilesource.TILE_FORMAT_NUMPY)):

    im_tile = np.array(slide_info['tile'])
    tile_mean_rgb = np.mean(im_tile[:, :, :3], axis=(0, 1))
    count_tot += 1
    visualize = True

    if np.mean(tile_mean_rgb) < 220. and im_tile.shape == (224, 224, 3):
        while count_tot > 2000 and count_cond < 10:
            count_cond += 1
            if visualize:
                plt.plot()
                plt.imshow(im_tile)
                plt.savefig("/CECI/home/users/m/d/mdausort/Cytology/temp/" + str(count_cond) + ".png")
