import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn import preprocessing

import time

np.random.seed(9)
seed = 1

dataset = np.loadtxt(r'training_data.csv', delimiter=',')

X = dataset[:, 0:-1]  # get the attributes
y = dataset[:, -1]  # get the classes

clf = RandomForestClassifier()

print("")
print("Validation:")
kf = KFold(n_splits=10)
print(kf)
print("")
print("Learning:")
print(clf)
print("=======")
print("Training on the training data:")

scores = []
fold = 0
for train_index, test_index in kf.split(X):
    print("Fold: " + str(fold))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    scores.append(acc)
    fold = fold + 1
acc = np.mean(scores)
std = np.std(scores)
print("Finished training via validation:")
print("Accuracy: " + str(acc * 100) + ", std: " + str(std * 100))

print("Fitting model to whole dataset...")
clf = RandomForestClassifier()
clf.fit(X, y)
fitted = clf.predict(X)
fitted_acc = accuracy_score(y, fitted)
print("Result was " + str(fitted_acc * 100))
print("This will always score 100%, since there is no validation at all.")

print("")
print(
    "Now, let's give the model some unseen data to emulate real time classification via the pre-trained model that we just created.")

unseen_data = np.loadtxt(r'unseen_data.csv', delimiter=',')
X_unseen = unseen_data[:, 0:-1]
y_unseen = unseen_data[:, -1]

realtime = clf.predict(X_unseen)
realtime_acc = accuracy_score(y_unseen, realtime)
print("We scored: " + str(realtime_acc * 100) + "% on the unseen trial")

print("")
print('Now we will "stream" this unseen data and predict them all')
print("This part is just to show you how to predict a single data object")
print(
    "Really, only the above result is needed, since it gives an overall view of how well the model performed on the new and unseen data")
print("But, if you do build a full-on real-time system, then a loop with the below code in would be useful")
input("Press Enter to start streaming and predicting...")

for data_object in unseen_data:
    X_single = data_object[0:-1].reshape(1, -1)
    y_single = data_object[-1]
    prediction = clf.predict(X_single)
    print("Predicted: " + str(prediction[0]) + " Actual: " + str(y_single))

print("")
print("Now that's done, sometimes we may not have any ground truth classes")
print("For example, if you're streaming straight from the Muse")
print(
    "See the comment in the below code if that is the case, because you just predict the data object and there's no accuracy since we don't know the true value")
input("Press Enter to start streaming and predicting...")

for data_object in unseen_data:
    X_single = data_object[0:-1].reshape(1, -1)
    prediction = clf.predict(X_single)
    # if there is no class column then you'd just do:  prediction = clf.predict(data_object)
    print("Predicted: " + str(prediction[0]))