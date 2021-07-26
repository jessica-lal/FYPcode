import numpy as np
from sklearn.ensemble import BaggingClassifier
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

# Create bagging classifier object
model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=10)

k_fold = KFold(n_splits=10)

print("")
print("The validation used in this experiment is:")
print(k_fold)
print("")
print("The classifier used in this experiment is:")
print(model)
print("=======")
print("Training performed on the training data:")

score_collection = []
fold_number = 0
for train_index, test_index in k_fold.split(X):
    print("Fold: " + str(fold_number))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    score_collection.append(accuracy)
    fold_number = fold_number + 1
accuracy = np.mean(score_collection)
standard_deviation = np.std(score_collection)
print("Finished training via validation:")
print("Accuracy: " + str(accuracy * 100) + ", Standard deviation: " + str(standard_deviation * 100))
