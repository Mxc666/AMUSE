from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier




def classifiers_func(X_train, Y_train, X_test, Y_test, SF):
    
    classifiers = {}
    
    X_train = X_train[SF]
    X_test = X_test[SF]
    
    
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    

    clf = svm.SVC()  # SVM
    clf.fit(X_train, Y_train)  
    
    kNN = KNeighborsClassifier(n_neighbors=3)   # kNN
    kNN.fit(X_train, Y_train)  
    
    gnb1 = GaussianNB() # Gaussian NB
    gnb1.fit(X_train, Y_train)
    
    gnb2 = MultinomialNB()  # nomial NB
    gnb2.fit(X_train, Y_train)
    
    # c45 = tree.DecisionTreeClassifier(criterion='entropy') # ID3
    # c45.fit(X_train, Y_train)
    #
    # bdt = AdaBoostClassifier(GaussianNB(), algorithm="SAMME", \
    #                          n_estimators=200, learning_rate=0.8) # AdaBoost
    # bdt.fit(X_train, Y_train)

    
    classifiers['SVM'] = clf
    classifiers['kNN'] = kNN
    classifiers['Gaussian NB'] = gnb1
    classifiers['Multinomial NB'] = gnb2
    # classifiers['C4.5'] = c45
    # classifiers['AdaBoost'] = bdt
    
    
    return classifiers, X_test, Y_test
        
