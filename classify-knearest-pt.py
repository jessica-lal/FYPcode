import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

training_dataset = np.loadtxt(r'training_data.csv', delimiter=',')

# obtain the attributes
X = training_dataset[:, 0:-1]

# obtain the classes
y = training_dataset[:, -1]

# manually change the number of neighbours from 1 to 25, in jumps of 1
neighbors = 25
classifier = KNeighborsClassifier(n_neighbors=neighbors)

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