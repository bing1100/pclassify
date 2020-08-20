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
from itertools import compress

def findOuterKPs(origin, hough, harris, bounds):
    l = []
    kps = []
    inters = []
    s = 0

    for i, (_, angle, dist) in enumerate(zip(*hough)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        p0 = np.asarray([origin[0], y0])
        p1 = np.asarray([origin[1], y1])
    
        _, angles, dists = hough 
        for angle, dist in zip(angles[i+1:], dists[i+1:]):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            p2 = np.asarray([origin[0], y0])
            p3 = np.asarray([origin[1], y1])
            
            inter = u.intersection(u.line(p0, p1), u.line(p2, p3))
            if inter and inter[0] < bounds[0] and inter[1] < bounds[1]:
                cand = u.idPoint(inter, harris, 1)
                if cand:
                    inters.append(inter)
                    kps.append([harris[cand[0]][1], harris[cand[0]][0]])
                    s += u.length(inter, [harris[cand[0]][1], harris[cand[0]][0]])

    thresh = s / len(inters)
    res = [kps[i] if thresh < u.length(kps[i], inters[i]) else inters[i] for i in range(4)]
    
    
    
    return np.array(res)

class KPMask():
    def __init__(self, resFile, labeled=True):
        f = open(resFile, "r")
        lines = (f.read()).split("\n")
        changes = 0
        if not labeled:
            lines = lines[:-1]
        
        for line in lines[:1]:
            name = ((line.split(","))[0].split("."))[0]
            sRes = float((line.split(","))[1])
            
            # Generate noisy image of a square
            imgFile = FILEROOT + "/trmask/"  + "1.tif"
            image = io.imread(imgFile, plugin='pil')
            thresh = threshold_otsu(image)
            bw = closing(image > thresh, square(3))
            cleared = clear_border(bw)
            label_image = label(cleared)
            
            for region in regionprops(label_image): 
                convex = region.convex_image
                mask = region.image
                
                convex = pad(convex, ((100,100),(100,100)))
                mask = pad(mask, ((100,100),(100,100)))
                
                median = np.median(np.where(mask), axis=1)
                mean = np.mean(np.where(mask), axis=1)

                origin = np.array((0, convex.shape[1]))
                harris = corner_peaks(corner_harris(mask, k=0), min_distance=3, threshold_rel=0)
                edges = feature.canny(convex, sigma=0.01)
                
                tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 90)
                h, theta, d = hough_line(edges, theta=tested_angles)
                hough = hough_line_peaks(h, theta, d, min_angle=25, num_peaks=4)
                print(hough[0].shape)
                outerKPs = findOuterKPs(origin, hough, harris, mask.shape)
                
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

                ax[0].imshow(edges, cmap=plt.cm.gray)
                ax[0].plot(harris[:, 1], harris[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
                ax[0].plot(mean[1], mean[0], color='blue', marker='o', linestyle='None', markersize=6)
                ax[0].plot(median[1], median[0], color='red', marker='o', linestyle='None', markersize=6)
                ax[0].axis('off')
                ax[0].set_title('convex', fontsize=20)
                
                ax[1].imshow(mask, cmap=plt.cm.gray)
                ax[1].plot(outerKPs[:, 0], outerKPs[:, 1], color='cyan', marker='o', linestyle='None', markersize=6)
                ax[1].plot(mean[1], mean[0], color='blue', marker='o', linestyle='None', markersize=6)
                ax[1].plot(median[1], median[0], color='red', marker='o', linestyle='None', markersize=6)
                ax[1].axis('off')
                ax[1].set_title('convex', fontsize=20)
                
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Loading Training Data...\n")
    tkps = KPMask("./trainres.txt")
    # print("Creating Training Stats...\n")
    # tkps.createStats()
    # print("Creating Training Box Plots...\n")
    # tkps.createBPs()
    # print("Exporting Training Data...\n")
    # tkps.exportData(TRAINAME)

    # print("Loading Validation Data...\n")
    # vkps = KPXML("./valres.txt", "./valxmls/", label=False)
    # print("Saving Validation Data...\n")
    # vkps.exportData(VALINAME)