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
from skimage.morphology import skeletonize

from skimage.morphology import skeletonize_3d
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt

import skimage.morphology
from scipy.spatial import ConvexHull

def get_endpoints_junction(mask_skeleton):
    w, h = mask_skeleton.shape
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    endpoints = []
    junctions = []
    for i in range(w):
        for j in range(h):
            if mask_skeleton[i,j] == 1:
                n = 0
                for ne in neigh:
                    ni = i+ne[0]
                    nj = j+ne[1]
                    if ni >= 0 and ni < w and nj >= 0 and nj < h:
                        if mask_skeleton[ni,nj] == 1:
                            n += 1
                if n == 1:
                    endpoints.append((i,j))
                elif n > 2:
                    junctions.append((i,j))
    
    # join junctions
    thre = 10
    if len(junctions) > 1:
        new_junctions = []
        Y = squareform(pdist(np.array(junctions), 'euclidean'))
        groups = np.zeros(Y.shape[0],)
        g = 1
        for i in range(Y.shape[0]):
            if groups[i] == 0:
                groups[Y[i,:] < thre] = g
                g += 1
        for g in np.unique(groups):
            pos = groups == g
            jun_g = np.array(junctions)[pos,:]
            #print(np.mean(jun_g, axis=0))
            new_junctions.append(np.mean(jun_g, axis=0))
        junctions = new_junctions
        # center and back
        if len(junctions) > 2 or len(junctions) == 0:
            print('Junctions: %d' % len(junctions))
        #	sys.exit(0)

        if len(junctions) == 2:
            dj = np.zeros(2,)
            for j in range(2):
                dj[j] = w*h
                for e in endpoints:
                    dej = (e[0]-junctions[j][0])*(e[0]-junctions[j][0]) + (e[1]-junctions[j][1])*(e[1]-junctions[j][1])
                    if dej < dj[j]:
                        dj[j] = dej
            if dj[0] > dj[1]:
                junctions = [junctions[0]]
            else:
                junctions = [junctions[1]]

    # join endpoints
    while len(endpoints) > 4:
        new_endpoints = []
        Y = pdist(np.array(endpoints), 'euclidean')
        l = np.argmin(Y)
        a, b = np.triu_indices(len(endpoints), k=1)
        for i, e in enumerate(endpoints):
            if i != a[l] and i != b[l]:
                new_endpoints.append(e)
        new_endpoints.append(((endpoints[a[l]][0] + endpoints[b[l]][0])/2., (endpoints[a[l]][1] + endpoints[b[l]][1])/2.))
        endpoints = new_endpoints
        
    endpoints = np.flip(endpoints)
    junctions = np.flip(junctions)

    return np.array(endpoints), np.array(junctions)

def tailSide(ps, i, j, cand):
    x, y = [c for c in range(4) if c not in [i, j]]
    l = [[u.length(ps[r], ps[t]) for r in [x, y]] for t in [i, j]]
    idx = np.argmax(l, axis=1)
    idx = np.argmin([l[0][idx[0]], l[1][idx[1]]])
    
    wing = [i,j][idx]
    
    return wing

def findKps(convex, mask, mcenter):
    bounds = mask.shape
    origin = np.array((0, convex.shape[1]))
    
    # Get Keypoints
    skeleton = skeletonize(mask)
    ep, cp = get_endpoints_junction(skeleton)
    kpmask = corner_peaks(corner_harris(mask, k=0), min_distance=3, threshold_rel=0)
    kpmask = np.flip(kpmask)
    
    # Get convex outline
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 90)
    edges = feature.canny(convex, sigma=0.01)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hough = hough_line_peaks(h, theta, d, min_angle=25, num_peaks=4)

    # Get the skeleton keypoints
    kpskel = corner_peaks(corner_harris(skeleton, k=0), min_distance=10, threshold_rel=0)
    kpskel = np.flip(kpskel)
    
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
            
            # Iterate through intersections
            inter = u.intersection(u.line(p0, p1), u.line(p2, p3))
            if inter and inter[0] < bounds[0] and inter[1] < bounds[1]:
                # Find all candidate points in kpmask corners
                cand = u.idPoint(inter, kpmask, 1)
                inters.append(inter)
                kps.append(cand[0])
                s += u.length(inter, cand[0])

    # Filter through whether to choose intersections or kpmask corners
    thresh = s / len(inters)
    res = [kps[i] if thresh < u.length(kps[i], inters[i]) else inters[i] for i in range(4)]
    h, t, ws = u.findOrientation(cp[0], res)
    
    cand = u.idPoint(h, kpskel, 1)
    h = u.getProjPoint(cand[0], mcenter, h)
    
    # Calculate center point using geometry and centroid
    cp = [
        (ws[0][0] + ws[1][0] + h[0] + mcenter[0])/4,
        (ws[0][1] + ws[1][1] + h[1] + mcenter[1])/4
    ]
    
    # Find the tail center using geometry
    ctp = u.getProjPoint(mcenter, h, t)
    cand = u.idPoint(ctp, kpskel, 1)
    ctp = u.getProjPoint(cand[0], h, t)
    shift = [ctp[0] - t[0], ctp[1] - t[1]]
    t2 = np.array([ctp[0] + shift[0], ctp[1] + shift[1]])
    ct = np.array([t, ctp, t2])
    
    res = np.array(res)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True)
    ax[0].imshow(convex, cmap=plt.cm.gray)
    ax[0].plot(kpmask[:, 0], kpmask[:, 1], color='cyan', marker='1', linestyle='None', markersize=3)
    ax[0].plot(res[:, 0], res[:, 1], color='red', marker='o', linestyle='None', markersize=6)
    ax[0].axis('off')
    ax[0].set_title('convex', fontsize=20)
    
    ax[1].imshow(mask, cmap=plt.cm.gray)
    ax[1].plot(res[:, 0], res[:, 1], color='cyan', marker='o', linestyle='None', markersize=6)
    ax[1].plot(res[-1, 0], res[-1, 1], color='red', marker='o', linestyle='None', markersize=6)
    ax[1].axis('off')
    ax[1].set_title('convex', fontsize=20)
    
    ax[2].imshow(skeleton, cmap=plt.cm.gray)
    ax[2].plot(mcenter[0], mcenter[1], color='red', marker='o', linestyle='None', markersize=6)
    ax[2].plot(cp[0], cp[1], color='cyan', marker='o', linestyle='None', markersize=6)
    ax[2].plot(kpskel[:, 0], kpskel[:, 1], color='white', marker='1', linestyle='None', markersize=6)
    ax[2].axis('off')
    ax[2].set_title('skeleton', fontsize=20)
    
    ax[3].imshow(mask, cmap=plt.cm.gray)
    ax[3].plot(h[0], h[1], color='blue', marker='o', linestyle='None', markersize=6)
    ax[3].plot(ct[:, 0], ct[:, 1], color='red', marker='o', linestyle='None', markersize=6)
    ax[3].plot(cp[0], cp[1], color='cyan', marker='o', linestyle='None', markersize=6)
    ax[3].plot(ws[:, 0], ws[:, 1], color='pink', marker='o', linestyle='None', markersize=6)
    ax[3].plot(kpskel[:, 0], kpskel[:, 1], color='black', marker='1', linestyle='None', markersize=6)
    ax[3].axis('off')
    ax[3].set_title('convex', fontsize=20)
    
    plt.tight_layout()
    plt.show()
    
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
            imgFile = FILEROOT + "/trmask/" + name + ".tif"
            image = io.imread(imgFile, plugin='pil')
            thresh = threshold_otsu(image)
            bw = closing(image > thresh, square(3))
            cleared = clear_border(bw)
            label_image = label(cleared)
            
            for region in regionprops(label_image): 
                # Get mask and convex hull with padding
                convex = region.convex_image
                mask = region.image
                padY, padX = mask.shape 
                convex = pad(convex, ((2*padX, 2*padX),(2*padY, 2*padY)))
                mask = pad(mask, ((2*padX, 2*padX),(2*padY, 2*padY)))
                
                # Calculate useful variables
                marea = region.area
                mcenter = region.local_centroid
                mcenter = np.array((mcenter[1] + 2*padY, mcenter[0] + 2*padX))

                outerKPs = findKps(convex, mask, mcenter)
                
                

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