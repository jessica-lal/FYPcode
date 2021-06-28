import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier

training_dataset = np.loadtxt(r'training_data.csv', delimiter=',')

# obtain the attributes
X = training_dataset[:, 0:-1]

# obtain the classes
y = training_dataset[:, -1]

random_forest = RandomForestClassifier()
decision_tree = DecisionTreeClassifier()
k_neighbours = KNeighborsClassifier()
naive_bayes = GaussianNB()
svc = LinearSVC()
logistic_reg = LogisticRegression()
gaussian_process = GaussianProcessClassifier()

# Create adaboost classifier object
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1)
# adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=random_forest, learning_rate=1)
# adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=decision_tree, learning_rate=1)
# -adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=k_neighbours, learning_rate=1)-
# -adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=naive_bayes, learning_rate=1)-
# -adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)-
# -adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=logistic_reg, learning_rate=1)-
# -adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=gaussian_process, learning_rate=1)-

k_fold = KFold(n_splits=10)

print("")
print("The validation used in this experiment is:")
print(k_fold)
print("")
print("The classifier used in this experiment is:")
print(adaboost)
print("=======")
print("Training performed on the training data:")

score_collection = []
fold_number = 0
for train_index, test_index in k_fold.split(X):
    print("Fold: " + str(fold_number))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    adaboost.fit(X_train, y_train)

    predictions = adaboost.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    score_collection.append(accuracy)
    fold_number = fold_number + 1
accuracy = np.mean(score_collection)
standard_deviation = np.std(score_collection)
print("Finished training via validation:")
print("Accuracy: " + str(accuracy * 100) + ", std: " + str(standard_deviation * 100))
