from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

def main():
    iris = datasets.load_iris()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'class_weight': ["balanced",{2:2}]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters, n_jobs=4)
    clf.fit(iris.data, iris.target)
    print(clf.best_params_)
    print(clf.best_score_)
    print(clf.n_jobs)

if __name__ == '__main__':
    main()