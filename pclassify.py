import numpy as np
import graphviz 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
from pathos.multiprocessing import ProcessingPool as Pool

# General Parameters
NUMPROCESS = 8
TESTSIZE = 0.1
SEED = 42
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
    "other": 9
}

# classifier Parameters
names = ["Nearest_Neighbors", "Gaussian_Process", "Decision_Tree", 
         "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA"]

nb = [798, 303, 98, 345, 1075, 456, 339, 268, 9, 1535, 0]
s = sum(nb)
nb = np.array(nb) / s

classifiers = [
    KNeighborsClassifier(10),
    GaussianProcessClassifier(warm_start=True, multi_class='one_vs_one', n_jobs=-1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(priors=nb),
    QuadraticDiscriminantAnalysis()
]

# Load data
f = open("./measureData/data.txt", 'r')
lines = (f.read()).split("\n")

features, labels = [], []
for line in lines[:-1]:
    segments = line.split(",")
    feature = [float(val) for val in segments[1:]]
    labels.append(CNUM[segments[0]])
    features.append(feature)

# Create the training and testing set from the data
fTrain, fTest, lTrain, lTest = train_test_split(features, labels, stratify=labels, random_state=SEED, test_size=TESTSIZE)

def runClassifier(idx):
    name, clf = names[idx], classifiers[idx]
    print('-----------\n Training {}...\n'.format(name))
    clf.fit(fTrain, lTrain)
    trainScore = clf.score(fTrain, lTrain)
    testScore = clf.score(fTest, lTest)
    line = '-----------\n{}\nTraining Acc: {}\nTesting Acc: {}\n'.format(name, trainScore, testScore)
    print(line)
    return line

p = Pool(NUMPROCESS)
lines = p.map(runClassifier, list(range(len(names))))

with open("./measureData/" + "pclassify_results.txt",'w') as target:
    target.writelines(lines)


