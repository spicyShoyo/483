import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

X = np.loadtxt("pcaData/trainingData.csv")
y = np.loadtxt("pcaData/label.csv")

perm = np.random.permutation(1934)
train_perm=perm[:1934]
test_perm=perm[:1934]

X_train=X[train_perm]
print "dimension of training data is "+str(X_train.shape)
y_train=y[train_perm]
print "dimension of training label is "+str(y_train.shape)

testData=np.loadtxt("pcaData/rawData.csv")
eigenvectors=np.loadtxt("pcaData/eigenvectors.csv")
print "dimension of eigenvectors is "+str(eigenvectors.shape)

X_test=np.dot(np.matrix(testData), np.matrix(eigenvectors))[test_perm]
print "dimension of test data is "+str(X_test.shape)
y_test=y[test_perm]
print "dimension of test label is "+str(y_test.shape)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


score = accuracy_score(y_test, predictions)
print "Accuracy score of the random forest: ", score
matrix = confusion_matrix(y_test, predictions)
print "Confusion matrix: \n", matrix
