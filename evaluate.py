from sklearn.metrics import accuracy_score, matthews_corrcoef


def Acc(Y, Y_pred):
    
    return accuracy_score(Y, Y_pred)

def MCC(Y, Y_pred):
    return matthews_corrcoef(Y, Y_pred)

def Metrics(classifiers, X, Y, ISmcc):
    metrics = {}
    
    Y = Y.astype('int')
    for clf_key in list(classifiers.keys()):
        Y_pred = classifiers[clf_key].predict(X)
        Y_pred = Y_pred.astype('int')
    
        Acc_score = Acc(Y, Y_pred)
        metrics[clf_key + '_Acc'] = Acc_score
        
        if ISmcc:    
            MCC_score = MCC(Y, Y_pred)
            metrics[clf_key + '_MCC'] = MCC_score
    
    return metrics