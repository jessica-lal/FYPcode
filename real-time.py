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
# classifier = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=7)

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
