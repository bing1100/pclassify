import os
import gc
import time as time
import numpy as np
import matplotlib.pyplot as plt
import util as u
import itertools
import pandas as pd
import plotly.express as px
import statistics as s
from scipy import ndimage as ndi
import xml.etree.ElementTree as ET
from paths import FILEROOT, SAVEROOT, TRAINAME, VALINAME
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
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import skimage.morphology
from pathos.multiprocessing import ProcessingPool as Pool

SHOWFIGURES = False
SAVEFIGURES = True
SHOW = False

NUMPROCESS = 12
CNUM = {
    "Boeing737": 0,
    "Boeing747": 1,
    "Boeing777": 2,
    "Boeing787": 3,
    "A220": 4,
    "A321": 5,
    "A330": 6,
    "A350": 7,
    "ARJ21": 8,
    "other": 9,
    "invalid": 10,
    "NA": 11
}

def i(string):
    switch = {
        "cp": 0,
        "kps": 3,
        "gtb": 4,
        "gtn": 3,
    }
    return switch[string]

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
    
    if len(candjuncs) == 0:
        junc = [mcenter]
    
    # If only 4 points, return these points
    if len(candkps) <= 4:
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
    
    return junc[0], np.array(combos[0][0])
    
    

def findKps(convex, mask, mcenter):
    maxY, maxX = mask.shape

    # Get Keypoints
    skeleton = skeletonize(mask)
    ep, candcp = get_endpoints_junction(skeleton)
    kpmask = corner_peaks(corner_harris(mask, k=0), min_distance=1, threshold_rel=0)
    kpmask = np.flip(kpmask)
    
    # Get convex outline
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    edges = feature.canny(convex, sigma=0.01)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hough = hough_line_peaks(h, theta, d, min_distance=5)

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
    
    if len(ckpconv) < 4:
        junc = [mcenter]
        if len(candcp) != 0:
            junc = u.idPoint(mcenter, candcp)
        
        if len(ep) < 4:
            return [], [], [], [], []
        cp, h, ct, w1, w2, _ = u.fixKeypoints(junc[0], ep, 10, 0.2, 5)
        ws = [w1, w2]
        ts = []
        return h, ws, ct, ts, cp
                
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

def solveLine(line, data, labeled):
    name = ((line.split(","))[0].split("."))[0]
    sRes = float((line.split(","))[1])
    
    # Generate noisy image of a square
    imgFile = FILEROOT + "/valmask/" + name + ".tif"
    if labeled:
        imgFile = FILEROOT + "/trmask/" + name + ".tif"
        labelFile = FILEROOT + "/label_xml/" + name + ".xml"
    
    image = io.imread(imgFile, plugin='pil')
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    cleared = clear_border(bw)
    label_image = label(cleared)
        
    d = []
    for region in regionprops(label_image): 
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

        pH, ws, pT, ts, pC = findKps(convex, mask, mcenter)
        
        plabel = "NA"
        if labeled and len(pH) != 0:
            lTree = ET.parse(labelFile)
            lRoot = lTree.getroot()
            plabel = "invalid"
            for cand in lRoot.iter("object"):
                coords = [
                    u.s2n(cand[i("gtb")][0].text),
                    u.s2n(cand[i("gtb")][1].text),
                    u.s2n(cand[i("gtb")][2].text),
                    u.s2n(cand[i("gtb")][3].text),
                ]
                if u.within(coords, np.flip(region.centroid)):
                    plabel = cand[i("gtn")][0].text

        d.append([pH, ws, pT, ts, pC, plabel, marea, sRes, name])
        
    gc.collect()
    return d
    
class KPMask():
    def __init__(self, resFile, labeled=True):
        self.counter = [0]*11
        self.data = {
            "Boeing737": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "Boeing747": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "Boeing777": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "Boeing787": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "A220": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "A321": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "A330": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "A350": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "ARJ21": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "other": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "invalid": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
            "NA": {"pArea":[], "wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[], "whAng":[], "whLen": [], "wtAng": [], "wtLen":[], "file": []},
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
        lines = lines[:-1]
        
        p = Pool(NUMPROCESS)
        res = p.map(lambda line: solveLine(line, self.data, labeled), lines)
        
        for ds in res:
            for [pH, ws, pT, ts, pC, plabel, marea, sRes, name] in ds:
                
                if len(pH) == 0:
                    continue
                
                pW1 = ws[0]
                pW2 = ws[1]

                wa = u.getAngle(pW1, pC, pW2) 
                wa = wa if wa < 180 else 360 - wa

                wha = u.getAngle(pW1, pH, pW2) 
                wha = wha if wha < 180 else 360 - wha

                wta = u.getAngle(pW1, pT, pW2) 
                wta = wta if wta < 180 else 360 - wta

                self.data[plabel]["pArea"].append(marea * sRes**sRes)
                self.data[plabel]["wLen"].extend(u.getWingLens(pW1, pC, pW2, sRes))
                self.data[plabel]["wSpan"].append(u.getWingSpan(pW1, pW2, sRes))
                self.data[plabel]["wAngle"].append(wa)
                self.data[plabel]["fH2C"].append(u.getH2C(pH, pC, sRes))
                self.data[plabel]["fC2T"].append(u.getC2T(pC, pT, sRes))
                self.data[plabel]["fH2T"].append(u.getH2T(pH, pT, sRes))
                self.data[plabel]["whAng"].append(wha)
                self.data[plabel]["whLen"].extend(u.getWingLens(pW1, pH, pW2, sRes))
                self.data[plabel]["wtAng"].append(wta)
                self.data[plabel]["wtLen"].extend(u.getWingLens(pW1, pT, pW2, sRes))
                self.data[plabel]["file"].append(name)
                self.counter[CNUM[plabel]] += 1
        
        # for line in lines:
        #     name = ((line.split(","))[0].split("."))[0]
        #     sRes = float((line.split(","))[1])
            
        #     # Generate noisy image of a square
        #     imgFile = FILEROOT + "/trmask/" + name + ".tif"
        #     image = io.imread(imgFile, plugin='pil')
        #     thresh = threshold_otsu(image)
        #     bw = closing(image > thresh, square(3))
        #     cleared = clear_border(bw)
        #     label_image = label(cleared)
            
        #     if labeled:
        #         imgFile = FILEROOT + "/valmask/" + name + ".tif"
        #         labelFile = FILEROOT + "/label_xml/" + name + ".xml"
            
        #     for region in regionprops(label_image): 
        #         # Get mask and convex hull with padding
        #         convex = region.convex_image
        #         mask = region.image
        #         padY, padX = mask.shape
        #         padY = padY 
        #         padX = padX 
        #         convex = pad(convex, ((padX, padX),(padY, padY)))
        #         mask = pad(mask, ((padX, padX),(padY, padY)))
                
        #         # Calculate useful variables
        #         marea = region.area
        #         mcenter = region.local_centroid
        #         mcenter = np.array((mcenter[1] + padY, mcenter[0] + padX))

        #         pH, ws, pT, ts, pC = findKps(convex, mask, mcenter)
                
        #         if len(pH) == 0:
        #             print("error")
        #             continue
                
        #         if labeled:
        #             lTree = ET.parse(labelFile)
        #             lRoot = lTree.getroot()
        #             plabel = "invalid"
        #             for cand in lRoot.iter("object"):
        #                 coords = [
        #                     u.s2n(cand[i("gtb")][0].text),
        #                     u.s2n(cand[i("gtb")][1].text),
        #                     u.s2n(cand[i("gtb")][2].text),
        #                     u.s2n(cand[i("gtb")][3].text),
        #                 ]
        #                 if u.within(coords, pC):
        #                     plabel = cand[i("gtn")][0].text
        #                     self.counter[CNUM[plabel]] += 1
                            
        #         pW1 = ws[0]
        #         pW2 = ws[1]

        #         wa = u.getAngle(pW1, pC, pW2) 
        #         wa = wa if wa < 180 else 360 - wa

        #         wha = u.getAngle(pW1, pH, pW2) 
        #         wha = wha if wha < 180 else 360 - wha

        #         wta = u.getAngle(pW1, pT, pW2) 
        #         wta = wta if wta < 180 else 360 - wta

        #         self.data[plabel]["pArea"].append(marea * sRes**sRes)
        #         self.data[plabel]["wLen"].extend(u.getWingLens(pW1, pC, pW2, sRes))
        #         self.data[plabel]["wSpan"].append(u.getWingSpan(pW1, pW2, sRes))
        #         self.data[plabel]["wAngle"].append(wa)
        #         self.data[plabel]["fH2C"].append(u.getH2C(pH, pC, sRes))
        #         self.data[plabel]["fC2T"].append(u.getC2T(pC, pT, sRes))
        #         self.data[plabel]["fH2T"].append(u.getH2T(pH, pT, sRes))
        #         self.data[plabel]["whAng"].append(wha)
        #         self.data[plabel]["whLen"].extend(u.getWingLens(pW1, pH, pW2, sRes))
        #         self.data[plabel]["wtAng"].append(wta)
        #         self.data[plabel]["wtLen"].extend(u.getWingLens(pW1, pT, pW2, sRes))
        #         self.data[plabel]["file"].append(name)
                
    def createStats(self):
        for l in self.data.keys():
            lines = []
            for feature in self.data[l].keys():
                if feature == "file":
                    continue
                if feature == "idx":
                    continue
                
                fData = self.data[l][feature]

                if len(fData) == 0:
                    continue

                lines.append("-----------------\n")
                lines.append("Feature: " + feature + "\n")
                lines.append("#Elements: " + str(len(fData)) + "\n")
                lines.append("Mean: " + str(s.mean(fData)) + "\n")
                lines.append("Median: " + str(s.median(fData)) + "\n")
                lines.append("STD: " + str(s.stdev(fData)) + "\n")
                lines.append("Var: " + str(s.variance(fData)) + "\n")
                lines.append("Quantiles: " + str(np.quantile(fData, [0.05,0.1,0.15,0.2,0.8,0.85,0.90,0.95])) + "\n")

                fig = px.histogram(fData, marginal="rug", title=self.features[feature][0])

                if SAVEFIGURES:
                    if not os.path.exists(SAVEROOT):
                        os.mkdir(SAVEROOT)

                    if not os.path.exists(SAVEROOT + l):
                        os.mkdir(SAVEROOT + l)

                    fig.write_image(SAVEROOT + l + "/" + str(feature) + "_hist.png")

                if SHOWFIGURES:
                    fig.show()
        
            if not os.path.exists(SAVEROOT):
                os.mkdir(SAVEROOT)

            if not os.path.exists(SAVEROOT + l):
                os.mkdir(SAVEROOT + l)

            with open(SAVEROOT + l + "/statistics.txt",'w') as target:
                target.writelines(lines)

    def createBPs(self):
        for feature in self.features.keys():
            if feature == "file":
                continue
            if feature == "idx":
                continue
            df = []
            for l in self.data.keys():
                for val in self.data[l][feature]:
                    df.append({"label": l, feature: val})
            
            df = pd.DataFrame.from_dict(data=df)

            title = feature + " Box Plot"

            fig = px.box(df, x="label", y=feature, points="all", title=title)
            if SAVEFIGURES:
                if not os.path.exists(SAVEROOT):
                    os.mkdir(SAVEROOT)
                fig.write_image(SAVEROOT + str(feature) + "_box_plot.png")

    def exportData(self, savename):
        lines = []
        for l in self.data.keys():
            nEle = len(self.data[l]["wSpan"])
            for idx in range(nEle):
                line = l
                for feature in self.data[l].keys():
                    if feature=="wLen" or feature=="whLen" or feature=="wtLen":
                        line += ',{},{}'.format(self.data[l][feature][idx], self.data[l][feature][nEle + idx - 1])
                    else:
                        line += ',{}'.format(self.data[l][feature][idx])
                line += "\n"
                lines.append(line)

        with open(SAVEROOT + savename,'w') as target:
            if not os.path.exists(SAVEROOT):
                os.mkdir(SAVEROOT)
            target.writelines(lines)

if __name__ == "__main__":
    print("Loading Training Data...\n")
    tkps = KPMask("./trainres.txt")
    print("Creating Training Stats...\n")
    tkps.createStats()
    print("Creating Training Box Plots...\n")
    tkps.createBPs()
    print("Exporting Training Data...\n")
    tkps.exportData(TRAINAME)

    print("Loading Validation Data...\n")
    vkps = KPMask("./valres.txt", labeled=False)
    print("Saving Validation Data...\n")
    vkps.exportData(VALINAME)
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