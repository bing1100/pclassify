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
    "other": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
    "invalid": {"wLen":[], "wSpan":[], "wAngle":[], "fH2C":[], "fC2T":[], "fH2T":[]},
}

features = {
    "wLen": ["Wing Lengths Histogram"],
    "wSpan": ["Wing Spans Histogram"],
    "wAngle": ["Wing Angle Histogram"],
    "fH2C": ["Head to Center Length Histogram"],
    "fC2T": ["Center to Tail Length Histogram"],
    "fH2T": ["Head to Tail Length Histogram"]
}

tRes = open("./trainres.txt", "r")



vRes = open("./valres.txt", "r")
