import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

training_dataset = np.loadtxt(r'training_data.csv', delimiter=',')

# obtain the attributes
X = training_dataset[:, 0:-1]

# obtain the classes
y = training_dataset[:, -1]

# classifier = RandomForestClassifier()
# classifier = DecisionTreeClassifier()
# classifier = KNeighborsClassifier()
# classifier = GaussianNB()
# classifier = LinearSVC()
# classifier = LogisticRegression()

# for gaussian process classifier:
kernel = 1.0 * RBF(1.0)
classifier = GaussianProcessClassifier(kernel=kernel, random_state=0)

print("")
print("The validation used in this experiment is:")
k_fold = KFold(n_splits=10)
print(k_fold)
print("")
print("The classifier used in this experiment is:")
print(classifier)
print("=======")
print("Training performed on the training data:")

score_collection = []
fold_number = 0
for train_index, test_index in k_fold.split(X):
    print("Fold: " + str(fold_number))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    score_collection.append(accuracy)
    fold_number = fold_number + 1
accuracy = np.mean(score_collection)
standard_deviation = np.std(score_collection)
print("Finished training via validation:")
print("Accuracy: " + str(accuracy * 100) + ", Standard Deviation: " + str(standard_deviation * 100))