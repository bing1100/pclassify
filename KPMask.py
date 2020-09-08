import time as time
import numpy as np
import matplotlib.pyplot as plt
import util as u
import itertools
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
from operator import itemgetter
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

SHOWFIGURES = False
SAVEFIGURES = True
SHOW = False

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
    
def verifyKps(candjuncs, candkps, mcenter):
    # Get the junction closest to the center
    junc = u.idPoint(mcenter, candjuncs)
    # If only 4 points, return these points
    if len(candjuncs) == 4:
        return junc[0], candkps
    
    # Generate all possible combinations
    combos = list(itertools.combinations(candkps, 4))
    
    # Sort the list based on the sum of distances from the center
    for i, combo in enumerate(combos):
        pts = [
            [combo[0][6], combo[0][7]],
            [combo[1][6], combo[1][7]],
            [combo[2][6], combo[2][7]],
            [combo[3][6], combo[3][7]],
        ]
        tl = u.length(pts[0], mcenter) 
        tl += u.length(pts[1], mcenter) 
        tl += u.length(pts[2], mcenter) 
        tl += u.length(pts[3], mcenter)
        combos[i] = [combos[i], tl]
    combos = sorted(combos, key=itemgetter(1))
    
    # Find the best candidate from the least distance to most distance
    for combo, tl in combos:
        # Check that the points are necessarily a fair distance apart
        twopts = list(itertools.combinations(combo, 2))
        angles = []
        for twopt in twopts:
            pt1 = [twopt[0][6], twopt[0][7]]
            pt2 = [twopt[1][6], twopt[1][7]]
            angle = u.getAngle(pt1, junc[0], pt2)
            angles.append(angle if angle < 180 else 360-angle)
        if True in [angle < 50 for angle in angles]:
            continue
        
        # Check the polygon contains the center point
        poly = [
            [combo[0][6], combo[0][7]],
            [combo[1][6], combo[1][7]],
            [combo[2][6], combo[2][7]],
            [combo[3][6], combo[3][7]],
        ]
        if not u.within(poly, junc[0]):
            continue
        
        # Passing these two check return the points
        return junc[0], np.array(combo)
    
    return junc[0], np.array(combo)
    
    

def findKps(convex, mask, mcenter):
    maxY, maxX = mask.shape

    # Get Keypoints
    skeleton = skeletonize(mask)
    _, candcp = get_endpoints_junction(skeleton)
    kpmask = corner_peaks(corner_harris(mask, k=0), min_distance=1, threshold_rel=0)
    kpmask = np.flip(kpmask)
    
    # Get convex outline
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    edges = feature.canny(convex, sigma=0.01)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hough = hough_line_peaks(h, theta, d, min_distance=10)

    # Get the skeleton keypoints
    kpskel = corner_peaks(corner_harris(skeleton, k=0), min_distance=1, threshold_rel=0)
    kpskel = np.flip(kpskel)
    
    ckpskel = []
    ckpmask = []
    ckpconv = []
    
    origin = np.array((0, mask.shape[1]))
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
            if inter and 0 < inter[0] and inter[0] < maxX and 0 < inter[1] and inter[1] < maxY:
                # Find all candidate points in kpmask corners
                ckpconv.append(inter)
                kp = u.idPoint(inter, kpskel)
                ckpskel.append(kp)
                kp = u.idPoint(inter, kpmask)
                ckpmask.append(kp)
                
                
    candkps = np.concatenate((np.array(ckpskel), np.array(ckpmask), np.array(ckpconv)), axis=1)
    cp, verikps = verifyKps(candcp, candkps, mcenter)

    # Filter through whether to choose intersections or kpmask corners
    thresh = np.mean(verikps[:,2])
    res = [verikps[i][3] if thresh/2 < verikps[i][2] else np.array([verikps[i][6], verikps[i][7]]) for i in range(4)]
    h, t, ws = u.findOrientation(cp, res)
    
    # Reorient Head
    cand = u.idPoint(h, kpskel)
    h = u.getProjPoint(cand[0], mcenter, h)
    
    # Calculate center point using geometry and centroid
    cp = [
        (ws[0][0] + ws[1][0] + h[0])/6 + cp[0]/2,
        (ws[0][1] + ws[1][1] + h[1])/6 + cp[1]/2
    ]
    
    # Find the tail center using geometry
    ct = u.getProjPoint(mcenter, h, t)
    cand = u.idPoint(ct, kpskel)
    ctp = u.getProjPoint(cand[0], h, t)
    shift = [ctp[0] - t[0], ctp[1] - t[1]]
    t2 = np.array([ctp[0] + shift[0], ctp[1] + shift[1]])
    ts = np.array([t, cand[0], t2])
    
    if SHOWFIGURES:
        res = np.array(res)
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True)
        ax[0].imshow(edges, cmap=plt.cm.gray)
        for _, angle, dist in zip(*hough):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[0].plot(origin, (y0, y1), '-r')
        ax[0].plot(kpmask[:, 0], kpmask[:, 1], color='cyan', marker='1', linestyle='None', markersize=3)
        ax[0].plot(res[:, 0], res[:, 1], color='red', marker='o', linestyle='None', markersize=6)
        ax[0].axis('off')
        ax[0].set_title('convex', fontsize=20)
        
        ax[1].imshow(mask, cmap=plt.cm.gray)
        ax[1].plot(res[:, 0], res[:, 1], color='cyan', marker='o', linestyle='None', markersize=6)
        ax[1].axis('off')
        ax[1].set_title('convex', fontsize=20)
        
        ax[2].imshow(skeleton, cmap=plt.cm.gray)
        ax[2].plot(mcenter[0], mcenter[1], color='red', marker='o', linestyle='None', markersize=6)
        ax[2].plot(candcp[0][0], candcp[0][1], color='cyan', marker='o', linestyle='None', markersize=6)
        ax[2].plot(kpskel[:, 0], kpskel[:, 1], color='white', marker='1', linestyle='None', markersize=6)
        ax[2].axis('off')
        ax[2].set_title('skeleton', fontsize=20)
        
        ax[3].imshow(mask, cmap=plt.cm.gray)
        ax[3].plot(h[0], h[1], color='blue', marker='o', linestyle='None', markersize=6)
        ax[3].plot(ct[0], ct[1], color='blue', marker='o', linestyle='None', markersize=6)
        ax[3].plot(ts[:, 0], ts[:, 1], color='red', marker='o', linestyle='None', markersize=6)
        ax[3].plot(cp[0], cp[1], color='cyan', marker='o', linestyle='None', markersize=6)
        ax[3].plot(ws[:, 0], ws[:, 1], color='pink', marker='o', linestyle='None', markersize=6)
        ax[3].plot(kpskel[:, 0], kpskel[:, 1], color='black', marker='1', linestyle='None', markersize=6)
        ax[3].axis('off')
        ax[3].set_title('convex', fontsize=20)
        
        plt.tight_layout()
        plt.show()
    
    return h, ws, ct, ts, cp

class KPMask():
    def __init__(self, resFile, labeled=True):
        self.data = {
            "Boeing737": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "Boeing747": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "Boeing777": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "Boeing787": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "A220": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "A321": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "A330": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "A350": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "ARJ21": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "other": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "invalid": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
            "NA": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": [], "idx": []},
        }

        self.features = {
            "pArea":["Area of the Plane Polygon Histogram"], 
            "wLen": ["Wing Lengths Histogram"],
            "wSpan": ["Wing Spans Histogram"],
            "wAngle": ["Wing Angle Histogram"],
            "fH2C": ["Head to Center Length Histogram"],
            "fC2T": ["Center to Tail Length Histogram"],
            "fH2T": ["Head to Tail Length Histogram"],
            "whAng": ["Wing Head Angle Histogram"],
            "whLen": ["Wing to Head Length Histogram"],
            "wtAng": ["Wing Tail Angle Histogram"],
            "wtLen": ["Wing to Tail Length Histogram"],
        }
        
        f = open(resFile, "r")
        lines = (f.read()).split("\n")
        changes = 0
        if not labeled:
            lines = lines[:-1]
        
        for line in lines[:5]:
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
                padY = padY 
                padX = padX 
                convex = pad(convex, ((padX, padX),(padY, padY)))
                mask = pad(mask, ((padX, padX),(padY, padY)))
                
                # Calculate useful variables
                marea = region.area
                mcenter = region.local_centroid
                mcenter = np.array((mcenter[1] + padY, mcenter[0] + padX))

                h, ws, ct, ts, cp = findKps(convex, mask, mcenter)

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