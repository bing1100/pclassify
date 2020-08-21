import os
import util as u
import xml.etree.ElementTree as ET
import plotly.express as px
import statistics as s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from paths import FILEROOT, SAVEROOT, TRAINAME, VALINAME

SHOWFIGURES = False
SAVEFIGURES = True
SHOW = False

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

class KPXML():
    def __init__(self, resFile, xmlroot, label=True):
        self.counter = [0]*11
        self.error = []
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
        if not label:
            lines = lines[:-1]
        
        for line in lines:
            name = ((line.split(","))[0].split("."))[0]
            sRes = float((line.split(","))[1])
            xmlFile = xmlroot + name + ".xml"

            if SHOW:
                imgFile = FILEROOT + "/images/" + name + ".tif"
                img=mpimg.imread(imgFile)
                plt.imshow(img)

            if label:
                labelFile = FILEROOT + "/label_xml/" + name + ".xml"

            tree = ET.parse(xmlFile)
            root = tree.getroot()

            idx = 0
            for plane in root.iter("object"):
                plabel = "NA"

                if len(list(plane[i("kps")])) != 4:
                    self.error.append("{} {}\n".format(name, idx))
                    continue

                kps = [
                    u.s2n(plane[i("kps")][0].text),
                    u.s2n(plane[i("kps")][1].text),
                    u.s2n(plane[i("kps")][2].text),
                    u.s2n(plane[i("kps")][3].text)
                ]

                if plane[i("cp")].text == "None":
                    pC = [
                        sum([i[0] for i in kps])/4,
                        sum([i[1] for i in kps])/4
                    ]
                else:
                    pC = u.s2n(plane[i("cp")].text)

                if SHOW:
                    xs = [i[0] for i in kps]
                    ys = [i[1] for i in kps]
                    plt.scatter(xs, ys)
                    plt.scatter(pC[0], pC[1])

                kps = u.fixKeypoints(pC, kps, 10, 0.2, 5)
                if SHOW:
                    xs = [i[0] for i in kps]
                    ys = [i[1] for i in kps]
                    plt.scatter(xs, ys)

                if label:
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
                        if u.within(coords, pC):
                            plabel = cand[i("gtn")][0].text
                            self.counter[CNUM[plabel]] += 1
                        else:
                            self.error.append(xmlFile)

                pC = kps[0]
                pH = kps[1]
                pT = kps[2]
                pW1 = kps[3]
                pW2 = kps[4]
                changes += kps[5]

                wa = u.getAngle(pW1, pC, pW2) 
                wa = wa if wa < 180 else 360 - wa

                wha = u.getAngle(pW1, pH, pW2) 
                wha = wha if wha < 180 else 360 - wha

                wta = u.getAngle(pW1, pT, pW2) 
                wta = wta if wta < 180 else 360 - wta

                self.data[plabel]["pArea"].append(u.getArea(kps[:-1], sRes))
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
                self.data[plabel]["idx"].append(str(idx))

                idx += 1

            if SHOW:
                plt.show()

        print("Counter: {}\n".format(str(self.counter)))
        print("Ps fixed: {}\n".format(str(changes)))

    def createStats(self):
        for label in self.data.keys():
            lines = []
            for feature in self.data[label].keys():
                if feature == "file":
                    continue
                if feature == "idx":
                    continue
                
                fData = self.data[label][feature]

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

                    if not os.path.exists(SAVEROOT + label):
                        os.mkdir(SAVEROOT + label)

                    fig.write_image(SAVEROOT + label + "/" + str(feature) + "_hist.png")

                if SHOWFIGURES:
                    fig.show()
        
            if not os.path.exists(SAVEROOT):
                os.mkdir(SAVEROOT)

            if not os.path.exists(SAVEROOT + label):
                os.mkdir(SAVEROOT + label)

            with open(SAVEROOT + label + "/statistics.txt",'w') as target:
                target.writelines(lines)

    def createBPs(self):
        for feature in self.features.keys():
            if feature == "file":
                continue
            if feature == "idx":
                continue
            df = []
            for label in self.data.keys():
                for val in self.data[label][feature]:
                    df.append({"label": label, feature: val})
            
            df = pd.DataFrame.from_dict(data=df)

            title = feature + " Box Plot"

            fig = px.box(df, x="label", y=feature, points="all", title=title)
            if SAVEFIGURES:
                if not os.path.exists(SAVEROOT):
                    os.mkdir(SAVEROOT)
                fig.write_image(SAVEROOT + str(feature) + "_box_plot.png")

    def exportData(self, savename):
        lines = []
        for label in self.data.keys():
            nEle = len(self.data[label]["wSpan"])
            for idx in range(nEle):
                line = label
                for feature in self.data[label].keys():
                    if feature=="wLen" or feature=="whLen" or feature=="wtLen":
                        line += ',{},{}'.format(self.data[label][feature][idx], self.data[label][feature][nEle + idx - 1])
                    else:
                        line += ',{}'.format(self.data[label][feature][idx])
                line += "\n"
                lines.append(line)

        with open(SAVEROOT + savename,'w') as target:
            if not os.path.exists(SAVEROOT):
                os.mkdir(SAVEROOT)
            target.writelines(lines)


if __name__ == "__main__":
    print("Loading Training Data...\n")
    tkps = KPXML("./trainres.txt", "./trainxmls/")
    print("Creating Training Stats...\n")
    tkps.createStats()
    print("Creating Training Box Plots...\n")
    tkps.createBPs()
    print("Exporting Training Data...\n")
    tkps.exportData(TRAINAME)

    print("Loading Validation Data...\n")
    vkps = KPXML("./valres.txt", "./valxmls/", label=False)
    print("Saving Validation Data...\n")
    vkps.exportData(VALINAME)