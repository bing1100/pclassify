import numpy as np
import graphviz 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from p_tqdm import p_map
from pathos.multiprocessing import ProcessingPool as Pool

# General Parameters
CLASSIFIERSUSED = [1,1,0,1,1,0,0,0,1,0]
DATAROOT = "./exp1/"
TRAINDATA = "traindata.txt"
VALIDATIONDATA = "valdata.txt"
NUMPROCESS = 8
TESTSIZE = 0.1
SEED = 42
TEST = False
PREDICT = True
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
NUMC = {
    0: "Boeing737",
    1: "Boeing747",
    2: "Boeing777",
    3: "Boeing787",
    4: "A220",
    5: "A321",
    6: "A330",
    7: "A350",
    8: "ARJ21",
    9: "other",
    10: "invalid",
    11: "NA"
}

counter = [0]*11

# classifier Parameters
names = ["Nearest_Neighbors", "Gaussian_Process_1v1", "Decision_Tree", 
         "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA", "Gradient Boosting", 
         "Gaussian_Process_1vrest", ]
classifiers = [
    KNeighborsClassifier(10, weights="distance", leaf_size=100, n_jobs=-1),
    GaussianProcessClassifier(warm_start=True, n_restarts_optimizer=5, max_iter_predict=500, multi_class='one_vs_one', n_jobs=-1),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_jobs=-1),
    MLPClassifier(hidden_layer_sizes=(100,50, 25, 50,100), solver='sgd', learning_rate="adaptive", max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
    GaussianProcessClassifier(warm_start=True, n_restarts_optimizer=5, max_iter_predict=500, n_jobs=-1),()
]

# Load Train data
f = open(DATAROOT + TRAINDATA, 'r')
lines = (f.read()).split("\n")

features, labels = [], []
for line in lines[:-1]:
    segments = line.split(",")[:-2]
    feature = [float(val) for val in segments[1:]]
    labels.append(CNUM[segments[0]])
    features.append(feature)

if TEST:
    # Create the training and testing set from the data
    fTrain, fTest, lTrain, lTest = train_test_split(features, labels, stratify=labels, random_state=SEED, test_size=TESTSIZE)

    # Generate how well the model trains 
    def runClassifier(idx):
        name, clf = names[idx], classifiers[idx]
        clf.fit(fTrain, lTrain)
        trainScore = clf.score(fTrain, lTrain)
        testScore = clf.score(fTest, lTest)
        line = '-----------\n{}\nTraining Acc: {}\nTesting Acc: {}\n'.format(name, trainScore, testScore)
        return line

    p = Pool(NUMPROCESS)
    lines = p.map(runClassifier, list(range(len(names))))

    with open(DATAROOT + "pclassify_results.txt",'w') as target:
        target.writelines(lines)

if PREDICT:
    # Load prediction data
    f = open(DATAROOT + TRAINDATA, 'r')
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
                return np.array(clf.predict_proba(vFeatures))
            else:
                return np.array(clf.predict(vFeatures))
        return []

    p = Pool(NUMPROCESS)
    probs = p_map(createPredictions, list(range(len(names))))

    data = {}
    i = 0
    for label in vLabels:
        d = np.array([0.0] * 11)
        for cprob in probs:
            if cprob != []:
                d += cprob[i]
        
        maxIdx = np.argmax(d)
        counter[maxIdx] += 1
        lStr = NUMC[maxIdx]
        if maxIdx < 10:
            if label[0] in data.keys():
                data[label[0]].append((label[1], lStr))
            else:
                data[label[0]] = [(label[1], lStr)]
        i += 1

    print(counter)

    for key in range(1, 1001):
        lines = []
        if str(key) in data.keys():
            for obj in data[str(key)]:
                lines.append("{},{}\n".format(obj[0], obj[1]))
        with open("./results/" + str(key) + ".txt",'w') as target:
            target.writelines(lines) 

