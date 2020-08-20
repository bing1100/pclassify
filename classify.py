import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
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
from sklearn.ensemble import VotingClassifier
from p_tqdm import p_map
from progress.bar import Bar
from pathos.multiprocessing import ProcessingPool as Pool
from paths import SAVEROOT, TRAINAME, VALINAME

# General Parameters
TEST = True
PREDICT = False
TESTINGDONE = [1,1,1,1,1,1,1,1,1,1,1,1,0]
CLASSIFIERSUSED = [0,0,0,0,0,1,1,0,0,0,1,0,1]
NUMPROCESS = 4
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

# classifier Parameters
names = [
    "Nearest_Neighbors",
    "Linear_SVM", 
    "RBF_SVM",
    "Gaussian_Process_1v1", 
    "Decision_Tree", 
    "Random_Forest", 
    "Neural_Net", 
    "AdaBoost",
    "Naive_Bayes", 
    "QDA", 
    "Gradient_Boosting", 
    "Gaussian_Process_1vrest",
    "Soft_Voting"
]

nb = [798, 303, 98, 345, 1075, 456, 339, 268, 9, 1535, 0]
s = sum(nb)
nb = np.array(sorted(nb, reverse=True)) / s

cused = sum(CLASSIFIERSUSED)

classifiers = [
    KNeighborsClassifier(10, weights="distance", n_jobs=NUMPROCESS),
    SVC(kernel="linear"),
    SVC(),
    GaussianProcessClassifier(multi_class='one_vs_one', n_jobs=NUMPROCESS),
    DecisionTreeClassifier(max_depth=6),
    RandomForestClassifier(max_depth=6, n_jobs=NUMPROCESS),
    MLPClassifier(hidden_layer_sizes=(100, 50, 25, 12, 25, 50, 100), solver='sgd', learning_rate="adaptive", max_iter=1000, n_iter_no_change=True),
    AdaBoostClassifier(n_estimators=200),
    GaussianNB(priors=nb),
    QuadraticDiscriminantAnalysis(priors=nb),
    GradientBoostingClassifier(max_depth=1, n_estimators=500, learning_rate=0.1),
    GaussianProcessClassifier(n_jobs=NUMPROCESS),
]

classifiers.append(
    VotingClassifier(
        estimators=[(names[idx], classifiers[idx]) for idx in [i for i, x in enumerate(CLASSIFIERSUSED[:-1]) if x == max(CLASSIFIERSUSED[:-1])]],
        voting='soft',
        n_jobs=NUMPROCESS
    )
)

# Load Train data
f = open(SAVEROOT + TRAINAME, 'r')
lines = (f.read()).split("\n")
from sklearn.model_selection import cross_validate
features, labels = [], []
for line in lines[:-1]:
    segments = line.split(",")[:-2]
    feature = [float(val) for val in segments[1:]]
    labels.append(CNUM[segments[0]])
    features.append(feature)

if TEST:
    # Generate how well the model trains 
    def runClassifier(idx):
        name, clf = names[idx], classifiers[idx]
        if TESTINGDONE[idx] == 1:
            cv_results = cross_validate(clf, features, labels, cv=ITER, return_train_score=True, n_jobs=NUMPROCESS)
            return [name, cv_results]
        return [name, -1]

    res = {}
    lines = []

    p = Pool(NUMPROCESS)
    scores = p_map(runClassifier, list(range(len(names))))

    for score in scores:
        n = score[0]
        r = score[1]

        line = '-----------\n{}\n'.format(n)
        if r != -1:
            trScore = r["train_score"]
            line += "Train Scores\n"
            line += 'Scores: {}\n'.format(trScore)
            line += "Accuracy: %0.2f (+/- %0.2f)\n" % (trScore.mean(), trScore.std())

            tScore = r["test_score"]
            line += "Train Scores\n"
            line += 'Scores: {}\n'.format(tScore)
            line += "Accuracy: %0.2f (+/- %0.2f)\n" % (tScore.mean(), tScore.std())
        lines.append(line)

    with open(SAVEROOT + "pclassify_results.txt",'w') as target:
        target.writelines(lines)

if PREDICT:
    # Load prediction data
    f = open(SAVEROOT + VALINAME, 'r')
    lines = (f.read()).split("\n")

    vFeatures, vLocations = [], []
    for line in lines[:-1]:
        segments = line.split(",")
        feature = [float(val) for val in segments[1:-2]]
        vFeatures.append(feature)
        vLocations.append([segments[-2], segments[-1]])

    def createPredictions(idx):
        if CLASSIFIERSUSED[idx] == 1:
            clf = classifiers[idx]
            clf.fit(features, labels)
            pred = clf.predict(vFeatures)
            
            return pred
        return []

    print("Prediction Phase\n")
    p = Pool(NUMPROCESS)
    probs = p_map(createPredictions, list(range(len(names))))

    m = [i for i, x in enumerate(CLASSIFIERSUSED) if x == max(CLASSIFIERSUSED)]
    idx = 0
    stats = []
    for prob in probs:
        if len(prob) == 0:
            continue
        algo = names[m[idx]]
        line = "----------------------------\n"
        line += algo + "\n"

        counter = [0]*11
        data = {}
        invCount = 0

        i = 0
        for loc in vLocations:
            c = prob[i]
            lStr = NUMC[c]
            counter[c] += 1
            if c < 10:
                if loc[0] in data.keys():
                    data[loc[0]].append((loc[1], lStr))
                else:
                    data[loc[0]] = [(loc[1], lStr)]
            else:
                invCount += 1
            i += 1
        
        line += "Invalid Cases Removed: " + str(invCount) + "\n"
        line += "Class Counter: " + str(counter) + "\n"

        print(line)
        stats.append(line)
        
        print("Generating Results\n")
        for key in range(1, 1001):
            lines = []
            if str(key) in data.keys():
                for obj in data[str(key)]:
                    lines.append("{},{}\n".format(obj[0], obj[1]))

            if not os.path.exists(SAVEROOT + "results_" + algo + "/"):
                os.mkdir(SAVEROOT + "results_" + algo + "/")
            
            with open(SAVEROOT + "results_" + algo + "/" + str(key) + ".txt",'w') as target:
                target.writelines(lines)
        
        idx += 1

    with open(SAVEROOT + "prediction_stats.txt",'w') as target:
        target.writelines(stats)

    print("Completed!\n")

