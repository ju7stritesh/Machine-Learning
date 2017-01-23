from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs
 
 
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
    numruns = 1

    classalgs = {
                 'Logistic Regression': algs.LogitReg(),
                 'Radial_Basis_Func': algs.Radial_Basis_Func({'k': 50, 's': 0.5})
                }
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 4},
        # {'regwgt': 0.01, 'nh': 8},
        # {'regwgt': 0.05, 'nh': 16},
        # {'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        # print r
    	trainset, testset = dtl.load_susy(trainsize,testsize)
        trainset1, testset1 = dtl.load_susy_complete(trainsize, testsize)
        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.iteritems():
                if (learnername == 'L1 Logistic Regression' or learnername == 'L2 Logistic Regression' or learnername == 'Logistic Alternative'):
                    train = trainset1
                    test = testset1
                else:
                    train = trainset
                    test = testset
                # Reset learner for new parameters
                learner.reset(params)
    	    	# print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
    	    	# Train model
    	    	learner.learn(train[0], train[1])
    	    	# Test model
    	    	predictions = learner.predict(test[0])
    	    	error = geterror(test[1], predictions)
                # print error
    	    	# print 'Error for ' + learnername + ': ' + str(error)
                errors[learnername][p,r] = error


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
