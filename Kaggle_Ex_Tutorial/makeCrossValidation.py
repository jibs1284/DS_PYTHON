__author__ = 'Joseba'

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import logloss
import numpy as np

def main():
    #Read in Data - Parse into Training - Target sets
    dataset = np.genfromtxt(open('Data/train.csv', 'r'), delimiter=',', dtype='f8')[1:]
    target = np.array(x[0] for x in dataset)
    train = np.array(x[1:] for x in dataset)

    #Use RandomForestClassifier
    cfr = RandomForestClassifier(n_estimators=100)

    #Simple K-Fold cross validation: 5 folds
    #Note: In older scikit-learn versions the n_folds argument is named k
    cv = cross_validation.KFold(len(train), n_folds=5, indices=False)

    #Iterate through Training - Test Cross Validation segments
    #Run Classifier on each one - Aggregate the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append(logloss.llfun(target[testcv], [x[1] for x in probas]))

    #Print out the mean of the cross-validated results
    print "Results: " + str(np.array(results).mean())

if __name__ == "__main__":
    main()