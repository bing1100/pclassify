from __future__ import division 
import math
from shapely.geometry import Point, Polygon
import numpy as np

# Lines (L*) are defined in the form Ax + By = C
# Points (p*) are defined in the tuple format (x, y)

def line(p1, p2):
    """
    Generates a line from two points
    :param p1, p2: two points
    :return: A, B, C line format
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = p1[0]*p2[1] - p2[0]*p1[1]
    return A, B, -C

def linePointSlope(L, p):
    """
    Generates a line with the same slope from a line and a point on the new line
    :param L, p: a line and a point
    :return: A, B, C line format
    """
    A = L[0]
    B = L[1]
    C = -L[1]*p[1] - L[0]*p[0]
    return A, B, -C

def linePointSlopeInverted(L, p):
    """
    Generates a line with the inverted slope from a line and a point on the new line
    :param L, p: a line and a point
    :return: A, B, C line format
    """
    A = -L[1]
    B = L[0]
    C = -L[0]*p[1] + L[1]*p[0]
    return A, B, -C

def length(p1, p2):
    """
    Calculates the euclidean distance between two points
    :param p1, p2: two points
    :return: the distance between the two lines
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def intersection(L1, L2):
    """
    Calculates the intersection between two lines
    :param L1, L2: two lines
    :return: a point that is the intersection or false if no point exists
    """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return False

def angle(p1, p2):
    """
    Calculates the angle of a line passing through two points
    :param p1, p2: two points on the line
    :return: the angle in degrees from 1 to 180 of the line
    """
    v = (p1[0] - p2[0], p1[1] - p2[1])
    a = math.degrees(math.atan2(v[1], v[0]))
    if a < 0:
        return 180 + a
    return 180 - a

def extend(p1, p2, val):
    """
    Calculates two new points further apart from each other by some value
    :param p1, p2, val: two points and a the value of extension
    :return: two new points ordered from left to right on the number line
    """
    s = p1
    e = p2
    if p2[0] < p1[0]:
        s = p2
        e = p1
    L = line(p1, p2)
    s[0] = s[0] - val
    e[0] = e[0] + val
    s[1] = -L[0]/L[1] * s[0] - L[2]/L[1]
    e[1] = -L[0]/L[1] * e[0] - L[2]/L[1]
    return s, e
    
def longer(cLong, cShort):
    """
    Determines the longer axis/line between 
    :param cLong, cShort: two tuple of tuples ((x1,y1), (x2,y2)) of a line segment
    :return: returns true if cLong is the longer segment, false otherwise
    """
    cLongMag = (cLong[0][0] - cLong[1][0])**2 + (cLong[0][1] - cLong[1][1])**2
    cShortMag = (cShort[0][0] - cShort[1][0])**2 + (cShort[0][1] - cShort[1][1])**2
    return cLongMag > cShortMag

def bucketCount(buckets, value, size):
    idx = int(min(value/size, len(buckets)-1))
    buckets[idx] += 1
    return buckets

def getWingAngle(pW1, pC, pW2):
    """
    Get the Angle created by 3 points
    :param p1, p2, p3: three tuples representing 3 points
    :return: returns the angle created by the 3 points in degrees
    """
    ang = math.degrees(math.atan2(pW2[1] - pC[1], pW2[0] - pC[0]) - math.atan2(pW1[1] - pC[1], pW1[0] - pC[0]))
    return ang + 360 if ang < 0 else ang

def dist2line(l1, l2, p3):
    return np.abs(np.cross(l2-l1, l1-p3)) / np.linalg.norm(l2-l1)

def getWingLens(pW1, pC, pW2, r):
    return length(pW1, pC) * r, length(pW2, pC) * r

def getWingSpan(pW1, pW2, r):
    return length(pW1, pW2) * r

def getH2C(pH, pC, r):
    return length(pH, pC) * r

def getC2T(pC, pT, r):
    return length(pC, pT) * r

def getH2T(pH, pT, r):
    return length(pH, pT) * r

def within(coords, candP):
    poly = Polygon(coords)
    cand = Point(candP)
    return cand.within(poly)

def getArea(coords, r):
    poly = Polygon(coords)
    return poly.area * r**2

def s2n(s):
    return [float(val) for val in s.split(",")]

def proj(u, v):  
    return (np.dot(u, v) / np.dot(v, v) ) * v 

def idLine(l1, l2, ps, num):
    cand = [-1,-1] * num

    for i in range(len(ps)):
        p = ps[i]
        l = dist2line(l1, l2, p)
        
        for ci in range(0, num+1, 2):
            if cand[ci + 1] == -1:
                cand[ci] = i
                cand[ci + 1] = l
                
            if l < cand[ci + 1]:
                ti = cand[ci]
                tl = cand[ci + 1]
                
                cand[ci] = i
                cand[ci + 1] = l
                
                i = ti
                l = tl
            
    return cand[::2]

def idPoint(p1, ps, num):
    cand = [-1,-1] * num
    
    for i in range(len(ps)):
        p = ps[i]
        p = (p[1], p[0])
        l = length(p1, p)
        
        for ci in range(0, num+1, 2):
            if cand[ci + 1] == -1:
                cand[ci] = i
                cand[ci + 1] = l
                
            if l < cand[ci + 1]:
                ti = cand[ci]
                tl = cand[ci + 1]
                
                cand[ci] = i
                cand[ci + 1] = l
                
                i = ti
                l = tl
    
    if cand[0] != -1:
        return cand[::2]
    
    return False

def findOrientation(pC, coords):
    diff = 180
    pos = [-1, -1]

    for i in range(4):
        for j in range(i+1, 4):
            ang = getWingAngle(coords[i], pC, coords[j])
            if abs(180 - ang) < diff:
                diff = abs(180 - ang)
                pos = [i, j]

    ws = [item for item in list(range(4)) if item not in pos]

    h = pos[0]
    t = pos[1]

    poly = [
        coords[pos[1]],
        coords[ws[0]],
        coords[ws[1]]
    ]
    if within(poly, pC):
        h = pos[1]
        t = pos[0]

    return h, t, ws

def fixKeypoints(pC, coords, wThreshold, fThreshold, aThreshold):
    h, t, ws = findOrientation(pC, coords)
    cPC = pC
    cH = coords[h]
    cT = coords[t]
    cWs = [coords[ws[0]], coords[ws[1]]]
    change = 0

    # Fix short head or tail
    hlen = length(cPC, cH)
    tlen = length(cPC, cT)
    diff = abs(hlen - tlen)

    if hlen < (fThreshold * tlen):
        change += 1
        v = [
            cH[0] - cPC[0],
            cH[1] - cPC[1]
        ]
        if v[1] == 0:
            cH[0] = cH[0] - np.sign(v[0]) * diff
        else:
            theta = math.atan(v[0] / v[1])
            cH = [
                cH[0] + diff * math.sin(theta),
                cH[1] + diff * math.cos(theta)
            ]
    if tlen < (fThreshold * hlen):
        change += 1
        v = [
            cT[0] - cPC[0],
            cT[1] - cPC[1]  
        ]
        if v[1] == 0:
            cT[0] = cT[0] - np.sign(v[0]) * diff
        else: 
            theta = math.atan(v[0] / v[1])
            cT = [
                cT[0] + diff * math.sin(theta),
                cT[1] + diff * math.cos(theta)
            ]

    # Fix short wings
    w1len = length(cPC, cWs[0])
    w2len = length(cPC, cWs[1])
    diff = abs(w1len - w2len)

    if wThreshold < diff:
        change += 1
        i = 1
        v = [
            cWs[1][0] - pC[0],
            cWs[1][1] - pC[1]
        ]
        if w1len < w2len:
            i = 0
            v = [
                cWs[0][0] - pC[0],
                cWs[0][1] - pC[1]
            ]
        if v[1] == 0:
            cWs[i][0] = cWs[i][0] + np.sign(v[0]) * diff
        else:
            theta = math.atan(v[0] / v[1])
            cWs[i] = [
                cWs[i][0] + diff * math.sin(theta),
                cWs[i][1] + diff * math.cos(theta)
            ]
    
    # Realign the plane center
    change += 1
    # cWp = np.array([
    #     (cWs[0][0] + cWs[1][0] + cH[0] + cPC[0])/4,
    #     (cWs[0][1] + cWs[1][1] + cH[1] + cPC[1])/4
    # ])
    # v = np.array([
    #      (cWs[0][1] - cWs[1][1]),
    #     -(cWs[0][0] - cWs[1][0])
    # ])
    # u = np.array([
    #     pC[0] - cWp[0],
    #     pC[1] - cWp[1]
    # ])
    # shift = proj(u, v)
    # cPC = [cWp[0] + shift[0], cWp[1] + shift[1]]
    # cPC = [cWp[0], cWp[1]]

    cPC = [
        (cWs[0][0] + cWs[1][0] + cH[0] + cPC[0])/4,
        (cWs[0][1] + cWs[1][1] + cH[1] + cPC[1])/4
    ]

    # Fix the tail angle to plane center and head
    angle = getWingAngle(cH, cPC, cT)
    if aThreshold < abs(180 - angle):
        change += 1
        v = np.array([
            cH[0] - cPC[0],
            cH[1] - cPC[1] 
        ])
        u = np.array([
            cT[0] - cPC[0],
            cT[1] - cPC[1]
        ])

        shift = proj(u, v)
        cT = [cPC[0] + shift[0], cPC[1] + shift[1]]

    return [cPC, cH, cT, cWs[0], cWs[1], change]

