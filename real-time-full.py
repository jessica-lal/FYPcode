import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
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

# ---------------------real-time-------------------
voting_one = RandomForestClassifier(n_estimators=50)
voting_two = KNeighborsClassifier()
voting_three = DecisionTreeClassifier()

classifier = RandomForestClassifier()
# classifier = RandomForestClassifier(n_estimators=50)
# classifier = KNeighborsClassifier()
# classifier = VotingClassifier(estimators=[('rf', voting_one), ('knn', voting_two), ('dt', voting_three)], voting='hard')
# classifier = VotingClassifier(estimators=[('rf', voting_one), ('knn', voting_two), ('dt', voting_three)], voting='soft')
# classifier = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=7), n_estimators=10)

print("The model " + str(classifier) + " will be fitted to the whole dataset")
classifier.fit(X, y)
fit = classifier.predict(X)
fit_accuracy = accuracy_score(y, fit)
print("The fitting accuracy score obtained is: " + str(fit_accuracy * 100)) # this will always score 100%, since there is no validation at all

# emulate real time classification via the pre-trained model created
print("")
print("Real time classification is emulated by giving the pre-trained model unseen data:")
unseen_dataset = np.loadtxt(r'unseen_data.csv', delimiter=',')
unseen_X = unseen_dataset[:, 0:-1]
unseen_y = unseen_dataset[:, -1]

realtime_classification = classifier.predict(unseen_X)
realtime_accuracy = accuracy_score(unseen_y, realtime_classification)
print("The real time classification emulation score obtained is: " + str(realtime_accuracy * 100))

#------------------------------------------------------------

print("")
print('Now we will "stream" this unseen data and predict them all')
print("This part is just to show you how to predict a single data object")
print(
    "Really, only the above result is needed, since it gives an overall view of how well the model performed on the new and unseen data")
print("But, if you do build a full-on real-time system, then a loop with the below code in would be useful")
input("Press Enter to start streaming and predicting...")

for data_object in unseen_dataset:
    X_single = data_object[0:-1].reshape(1, -1)
    y_single = data_object[-1]
    prediction = classifier.predict(X_single)
    print("Predicted: " + str(prediction[0]) + " Actual: " + str(y_single))

print("")
print("Now that's done, sometimes we may not have any ground truth classes")
print("For example, if you're streaming straight from the Muse")
print(
    "See the comment in the below code if that is the case, because you just predict the data object and there's no accuracy since we don't know the true value")
input("Press Enter to start streaming and predicting...")

for data_object in unseen_dataset:
    X_single = data_object[0:-1].reshape(1, -1)
    prediction = classifier.predict(X_single)
    # if there is no class column then you'd just do:  prediction = clf.predict(data_object)
    print("Predicted: " + str(prediction[0]))