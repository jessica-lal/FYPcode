import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50)
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier()
clf5 = LinearSVC()
clf6 = KNeighborsClassifier(n_neighbors=5)

#dataset = np.loadtxt(r'training_data.csv', delimiter=',')

#X = dataset[:,0:-1] #get the attributes
#y = dataset[:,-1] #get the classes


# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3), ('dt', clf4), ('svc', clf5), ('kn', clf6)],
        voting='hard')
eclf1 = eclf1.fit(X, y)
print(eclf1.predict(X))

#np.array_equal(eclf1.named_estimators_.lr.predict(X),
#               eclf1.named_estimators_['lr'].predict(X))

# eclf2 = VotingClassifier(estimators=[
#        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#        voting='soft')
# eclf2 = eclf2.fit(X, y)
# print(eclf2.predict(X))

# eclf3 = VotingClassifier(estimators=[
#       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#      voting='soft', weights=[2,1,1],
#       flatten_transform=True)
# eclf3 = eclf3.fit(X, y)
# print(eclf3.predict(X))

# print(eclf3.transform(X).shape)