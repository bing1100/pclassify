import numpy as np
import matplotlib.pyplot as plt
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

# Generate noisy image of a square
imgFile = FILEROOT + "/trmask/" + "5.tif"
im = io.imread(imgFile, plugin='pil')

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=0.01)
edges2 = feature.canny(im, sigma=3)

lines = probabilistic_hough_line(edges1, threshold=10, line_length=1,
                                 line_gap=1)

fd, hog_image = hog(im, orientations=9, pixels_per_cell=(4, 4),
                    visualize=True)

# display results
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(8, 3),
                                    sharex=True, sharey=True)

coords = corner_peaks(corner_harris(im), min_distance=5, threshold_rel=0.02)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title(r'$\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title(r'$\sigma=3$', fontsize=20)

ax4.imshow(edges2 * 0, cmap=plt.cm.gray)
for line in lines:
    p0, p1 = line
    ax4.plot((p0[0], p1[0]), (p0[1], p1[1]))
ax4.axis('off')
ax4.set_title(r'Hough', fontsize=20)

ax5.imshow(edges1, cmap=plt.cm.gray)
ax5.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)

plt.show()