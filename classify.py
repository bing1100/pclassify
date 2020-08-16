import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from p_tqdm import p_map
from progress.bar import Bar
from pathos.multiprocessing import ProcessingPool as Pool
from paths import SAVEROOT, TRAINAME, VALINAME

# General Parameters
TEST = True
PREDICT = False
TESTINGDONE = [0,1,1,0,1,1,1,0,0,0,1,0]
CLASSIFIERSUSED = [0,0,0,0,0,1,1,0,0,0,1,0]
NUMPROCESS = 8
TESTSIZE = 0.2
ITER = 20

CNUM = {
    "Boeing737": 1,
    "Boeing747": 5,
    "Boeing777": 7,
    "Boeing787": 3,
    "A220": 0,
    "A321": 2,
    "A330": 4,
    "A350": 6,
    "ARJ21": 8,
    "other": 9,
    "invalid": 10,
    "NA": 11
}

NUMC = {
    0: "A220",
    1: "Boeing737",
    2: "A321",
    3: "Boeing787",
    4: "A330",
    5: "Boeing747",
    6: "A350",
    7: "Boeing777",
    8: "ARJ21",
    9: "other",
    10: "invalid",
    11: "NA"
}

counter = [0]*11

# classifier Parameters
names = [
    "Nearest_Neighbors",
    "Linear SVM", 
    "RBF SVM",
    "Gaussian_Process_1v1", 
    "Decision_Tree", 
    "Random_Forest", 
    "Neural_Net", 
    "AdaBoost",
    "Naive_Bayes", 
    "QDA", 
    "Gradient Boosting", 
    "Gaussian_Process_1vrest", 
]

nb = [798, 303, 98, 345, 1075, 456, 339, 268, 9, 1535, 0]
s = sum(nb)
nb = np.array(sorted(nb, reverse=True)) / s

cused = sum(CLASSIFIERSUSED)

classifiers = [
    KNeighborsClassifier(10, weights="distance", n_jobs=-1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(multi_class='one_vs_one', n_jobs=-1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_jobs=-1),
    MLPClassifier(hidden_layer_sizes=(100, 50, 25, 12, 25, 50, 100), solver='sgd', learning_rate="adaptive", max_iter=1000, n_iter_no_change=True),
    AdaBoostClassifier(n_estimators=200),
    GaussianNB(priors=nb),
    QuadraticDiscriminantAnalysis(priors=nb),
    GradientBoostingClassifier(max_depth=1, n_estimators=300),
    GaussianProcessClassifier(n_jobs=-1),
]

# Load Train data
f = open(SAVEROOT + TRAINAME, 'r')
lines = (f.read()).split("\n")

features, labels = [], []
for line in lines[:-1]:
    segments = line.split(",")[:-2]
    feature = [float(val) for val in segments[1:]]
    labels.append(CNUM[segments[0]])
    features.append(feature)

if TEST:
    # Create the training and testing set from the data
    fTrain, fTest, lTrain, lTest = train_test_split(features, labels, stratify=labels, test_size=TESTSIZE)

    # Generate how well the model trains 
    def runClassifier(idx):
        name, clf = names[idx], classifiers[idx]
        if TESTINGDONE[idx] == 1:
            clf.fit(fTrain, lTrain)
            trainScore = clf.score(fTrain, lTrain)
            testScore = clf.score(fTest, lTest)
            line = '-----------\n{}\nTraining Acc: {}\nTesting Acc: {}\n'.format(name, trainScore, testScore)
            return [name, trainScore, testScore, line]
        return [name, -1, -1, "N/A\n"]

    res = {}
    lines = []

    bar = Bar("Testing Phase", max=ITER)
    for i in range(ITER):
        p = Pool(NUMPROCESS)
        data = p.map(runClassifier, list(range(len(names))))

        for i in data:
            k = i[0]
            trScore = i[1]
            tScore = i[2]
            line = i[3]

            if k not in res.keys():
                res[k] = [trScore, tScore]
            else:
                res[k][0] += trScore
                res[k][1] += tScore

            lines.append(line)

        bar.next()
    bar.finish()

    for k in res.keys():
        lines.append('-----------\n{}\nAveraged Training Acc: {}\n Averaged Testing Acc: {}\n'.format(k, res[k][0]/ITER, res[k][1]/ITER))

    with open(SAVEROOT + "pclassify_results.txt",'w') as target:
        target.writelines(lines)

if PREDICT:
    # Load prediction data
    f = open(SAVEROOT + VALINAME, 'r')
    lines = (f.read()).split("\n")

    vFeatures, vLabels = [], []
    for line in lines[:-1]:
        segments = line.split(",")
        feature = [float(val) for val in segments[1:-2]]
        vFeatures.append(feature)
        vLabels.append([segments[-2], segments[-1]])

    def createPredictions(idx):
        if CLASSIFIERSUSED[idx] == 1:
            clf = classifiers[idx]
            clf.fit(features, labels)
            if names[idx] != "Gaussian_Process_1v1":
                return clf.predict_proba(vFeatures)
            else:
                pred = clf.predict(vFeatures)
                pred = pred.reshape(len(pred), 1)
                enc = preprocessing.OneHotEncoder(sparse=False)
                onehotlabels = enc.fit_transform(pred)
                return onehotlabels
        return []

    print("Prediction Phase\n")
    p = Pool(NUMPROCESS)
    probs = p_map(createPredictions, list(range(len(names))))

    data = {}
    i = 0
    invCount = 0
    for label in vLabels:
        d = np.array([0.0] * 11)
        for cprob in probs:
            if len(cprob) != 0:
                d += cprob[i]

        # maxIdx = np.argwhere(d > cused * 0.50)
        # if maxIdx.size != 0:
        #     maxIdx = maxIdx = maxIdx[0][0]
        # else:
        #     maxIdx = np.argmax(d)

        maxIdx = np.argmax(d)

        counter[maxIdx] += 1
        lStr = NUMC[maxIdx]
        if maxIdx < 10:
            if label[0] in data.keys():
                data[label[0]].append((label[1], lStr))
            else:
                data[label[0]] = [(label[1], lStr)]
        else:
            invCount += 1
        i += 1
    
    print("Invalid Cases Removed: ", invCount, "\n")
    print("Class Counter: ", counter, "\n")

    print("Generating Results\n")
    for key in range(1, 1001):
        lines = []
        if str(key) in data.keys():
            for obj in data[str(key)]:
                lines.append("{},{}\n".format(obj[0], obj[1]))

        if not os.path.exists(SAVEROOT + "results/"):
            os.mkdir(SAVEROOT + "results/")
        
        with open(SAVEROOT + "results/" + str(key) + ".txt",'w') as target:
            target.writelines(lines)

    print("Completed!\n")

