import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
    training_dataset = np.loadtxt(r'training_data.csv', delimiter=',')
    # obtain the attributes
    X = training_dataset[:, 0:-1]
    # obtain the classes
    y = training_dataset[:, -1]
    return X, y

# get a voting ensemble of models
def get_voting():
    # define the base models
    models = list()
    models.append(('random-forest', RandomForestClassifier(n_estimators=50)))
    models.append(('k-nearest-neighbours', KNeighborsClassifier()))
    models.append(('decision-tree', DecisionTreeClassifier()))
    #models.append(('linear-svc', LinearSVC()))
    #models.append(('logistic-regression', LogisticRegression()))
    models.append(('gaussian-naive-bayes', GaussianNB()))
    models.append(('gaussian-process', GaussianProcessClassifier()))
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    #ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble

# get a list of models to evaluate
def get_models():
    models = dict()
    models['random-forest'] = RandomForestClassifier()
    models['k-nearest-neighbours'] = KNeighborsClassifier()
    models['decision-tree'] = DecisionTreeClassifier()
    #models['linear-svc'] = LinearSVC()
    #models['logistic-regression'] = LogisticRegression()
    models['gaussian-naive-bayes'] = GaussianNB()
    models['gaussian-process'] = GaussianProcessClassifier()
    models['hard_voting'] = get_voting()
    #models['soft_voting'] = get_voting()
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = KFold(n_splits=10)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    #print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    print(name, (np.mean(scores)*100), (np.std(scores)*100))
# plot model performance for comparison
#pyplot.boxplot(results, labels=names, showmeans=True)
#pyplot.show()