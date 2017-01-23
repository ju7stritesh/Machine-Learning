from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from scipy import stats
import dataloader as dtl
import classalgorithms as algs
from sklearn import model_selection
from sklearn.model_selection import KFold
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

 
if __name__ == '__main__':
    trainsize = 874
    testsize = 548
    numruns = 2

    classalgs = {
                 'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 'Logistic Regression': algs.LogitReg(),
                 'Neural Network': algs.NeuralNet({'epochs': 100})
                }
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 4},
        {'regwgt': 0.01, 'nh': 8},
        {'regwgt': 0.05, 'nh': 16},
        {'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
    	trainset, testset = dtl.load_banknote(trainsize,testsize)
        train_features=np.vstack((trainset[0],testset[0]))
        train_label = list(trainset[1]) +list(testset[1])
        train_label=np.array(train_label)
        k_data = model_selection.KFold(n_splits=3) #creating K folds

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.iteritems():
                error_list = []
                for trainset, testset in k_data.split(train_features):
                    train = trainset
                    test = testset
                    learner.reset(params)
                    learner.learn(train_features[train], train_label[train])
                    # Test model
                    predictions = learner.predict(train_features[test])
                    error = geterror(train_label[test], predictions)
                    error_list.append(error)
                    # print 'Error for ' + learnername + ': ' + str(error)
                errors[learnername][p,r] = np.mean(error_list)

    best_algo = {}
    for learnername, learner in classalgs.iteritems():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
    	print 'Best parameters for ' + learnername + ': ' + str(learner.getparams())
    	print 'Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(1.96*np.std(errors[learnername][bestparams,:])/math.sqrt(numruns))
        best_algo[learnername] = list(errors[learnername][bestparams,:])

    tvalue, pvalue = stats.ttest_ind(best_algo['Logistic Regression'], best_algo['Neural Network'])
    print "tvalue, pvalue",tvalue,pvalue
    threshold = .05
    if pvalue < threshold:
        print "The Null hypothesis is rejected"
    else:
        print "Cannot reject the Null hypothesis"