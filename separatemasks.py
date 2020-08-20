import time as time
import numpy as np
import matplotlib.pyplot as plt
import util as u
from scipy import ndimage as ndi
from paths import FILEROOT, SAVEROOT
from skimage import feature, io
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line
from skimage.feature import hog
from skimage import exposure
from skimage.feature import (
    corner_fast,
    corner_harris,
    corner_subpix, 
    corner_peaks
)
import matplotlib.patches as mpatches
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.util import pad

# Generate noisy image of a square
imgFile = FILEROOT + "/trmask/" + "304.tif"
image = io.imread(imgFile, plugin='pil')

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

print(cleared)

# label image regions
label_image = label(cleared)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

for region in regionprops(label_image):    
    convex = region.convex_image
    mask = region.image
    idx = convex != mask
    inverse = np.zeros(mask.shape)
    inverse[idx] = convex[idx]
    
    convex = pad(convex, ((5,5),(5,5)))
    mask = pad(mask, ((5,5),(5,5)))
    inverse = pad(inverse, ((5,5),(5,5)))
    
    # harris = corner_peaks(corner_harris(convex, k=0), min_distance=5, threshold_rel=0)
    harris = corner_peaks(corner_harris(mask, k=0), min_distance=3, threshold_rel=0)
    edges1 = feature.canny(convex, sigma=0.01)
    edges2 = feature.canny(mask, sigma=0.01)
    
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    h, theta, d = hough_line(edges1, theta=tested_angles)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    
    fig, (ax1, ax2, ax4) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
    origin = np.array((0, convex.shape[1]))
    points = set()
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        l1 = np.asarray([origin[0], y0])
        l2 = np.asarray([origin[1], y1])
        
        for _, angle1, dist1 in zip(*hough_line_peaks(h, theta, d)):
            y2, y3 = (dist1 - origin * np.cos(angle1)) / np.sin(angle1)
            l3 = np.asarray([origin[0], y2])
            l4 = np.asarray([origin[1], y3])
                 
            L1 = u.line(l1, l2)
            L2 = u.line(l3, l4)
            
            ax1.plot(origin, (y0, y1), '-b')
            ax1.plot(origin, (y2, y3), '-b')
            
            inter = u.intersection(L1, L2)
        
            if inter:
                ax1.plot(inter[0], inter[1], color='green', marker='o', linestyle='None', markersize=6)
                cand = u.idPoint(inter, harris, 1)
                
                if cand:
                    points.update(cand)
    
    ps = list(points)
    ps = np.array([harris[i] for i in list(points)])
    
    ax1.imshow(convex, cmap=plt.cm.gray)
    ax1.plot(harris[:, 1], harris[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
    ax1.plot(ps[:, 1], ps[:, 0], color='red', marker='o', linestyle='None', markersize=6)
    ax1.axis('off')
    ax1.set_title('convex', fontsize=20)
    
    ax2.imshow(edges2, cmap=plt.cm.gray)
    ax2.plot(harris[:, 1], harris[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
    ax2.set_xlim(origin)
    ax2.set_ylim((convex.shape[0], 0))
    ax2.axis('off')
    ax2.set_title('filled', fontsize=20)
    
    origin = np.array((0, convex.shape[1]))
    ax4.imshow(edges1, cmap=plt.cm.gray)
    ax4.plot(harris[:, 1], harris[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax4.plot(origin, (y0, y1), '-r')
    ax4.set_xlim(origin)
    ax4.set_ylim((convex.shape[0], 0))
    ax4.set_axis_off()
    ax4.set_title('Detected lines')

plt.tight_layout()
plt.show()