import os
import util as u
import xml.etree.ElementTree as ET
import plotly.express as px
import statistics as s
import numpy as np
import pandas as pd

FILEROOT = "/media/bhux/ssd/airplane/train/train"
SHOWFIGURES = False
SAVEFIGURES = True

data = {
    "Boeing737": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "Boeing747": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "Boeing777": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "Boeing787": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "A220": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "A321": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "A330": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "A350": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "ARJ21": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "other": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]}
}

features = {
    "wLen": ["Wing Lengths Histogram"],
    "wSpan": ["Wing Spans Histogram"],
    "wAngle": ["Wing Angle Histogram"],
    "fH2C": ["Head to Center Length Histogram"],
    "fC2T": ["Center to Tail Length Histogram"],
    "fH2T": ["Head to Tail Length Histogram"]
}

f = open("./resolution.txt", "r")
lines = (f.read()).split("\n")

# Generate Features From Keypoints
for line in lines:
    name = ((line.split(","))[0].split("."))[0]
    sRes = float((line.split(","))[1])

    kpsFile = FILEROOT + "/keypoints/" + name + ".txt"
    xmlFile = FILEROOT + "/label_xml/" + name + ".xml"

    kpsf = open(kpsFile, "r")
    kpsLines = (kpsf.read()).split("\n")
    kps = [line.split(",") for line in kpsLines[:-1]]

    tree = ET.parse(xmlFile)
    root = tree.getroot()

    idx = 0
    for name in root.iter("name"):
        label = name.text
        kp = kps[idx]

        xs = [int(float(i)) for i in kp[0::2]]
        ys = [int(float(i)) for i in kp[1::2]]

        pC = (xs[0], ys[0])
        pH = (xs[1], ys[1])
        pT = (xs[2], ys[2])
        pW1 = (xs[3], ys[3])
        pW2 = (xs[4], ys[4])

        data[label]["wLen"].extend(u.getWingLens(pW1, pC, pW2, sRes))
        data[label]["wSpan"].append(u.getWingSpan(pW1, pW2, sRes))
        data[label]["wAngle"].append(360 - u.getAngle(pW1, pC, pW2))
        data[label]["fH2C"].append(u.getH2C(pH, pC, sRes))
        data[label]["fC2T"].append(u.getC2T(pC, pT, sRes))
        data[label]["fH2T"].append(u.getH2T(pH, pT, sRes))
        
        idx += 1

# Create Statistics and Histograms for each plane class
for label in data.keys():
    lines = []
    for feature in data[label].keys():
        fData = data[label][feature]

        lines.append("-----------------\n")
        lines.append("Feature: " + feature + "\n")
        lines.append("#Elements: " + str(len(fData)) + "\n")
        lines.append("Mean: " + str(s.mean(fData)) + "\n")
        lines.append("Median: " + str(s.median(fData)) + "\n")
        lines.append("STD: " + str(s.stdev(fData)) + "\n")
        lines.append("Var: " + str(s.variance(fData)) + "\n")
        lines.append("Quantiles: " + str(np.quantile(fData, [0.05,0.1,0.15,0.2,0.8,0.85,0.90,0.95])) + "\n")

        fig = px.histogram(fData, marginal="rug", title=features[feature][0])

        if SAVEFIGURES:
            if not os.path.exists("./measureData/" + label):
                os.mkdir("./measureData/" + label)
            fig.write_image("./measureData/" + label + "/" + str(feature) + "_hist.png")

        if SHOWFIGURES:
            fig.show()
    
    with open("./measureData/" + label + "/statistics.txt",'w') as target:
        target.writelines(lines)

# Create Box Plots for comparison between classes
for feature in features.keys():
    df = []
    for label in data.keys():
        for val in data[label][feature]:
            df.append({"label": label, feature: val})
    
    df = pd.DataFrame.from_dict(data=df)

    title = feature + " Box Plot"

    fig = px.box(df, x="label", y=feature, points="all", title=title)
    if SAVEFIGURES:
        fig.write_image("./measureData/" + str(feature) + "_box_plot.png")

# Export Data to text file for further processing
lines = []
for label in data.keys():
    nEle = len(data[label]["wSpan"])
    for idx in range(nEle):
        line = label
        for feature in data[label].keys():
            if feature=="wLen":
                line += ',{},{}'.format(data[label][feature][idx], data[label][feature][nEle + idx - 1])
            else:
                line += ',{}'.format(data[label][feature][idx])
        line += "\n"
        lines.append(line)

with open("./measureData/" + "data.txt",'w') as target:
    target.writelines(lines)

