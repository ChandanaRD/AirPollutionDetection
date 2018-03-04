# data set display - head and tail
# description
# plots- buttons


# Load libraries
from sklearn import preprocessing, cross_validation, neighbors
import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
#url = "AirQualityUCI1.csv"
#names = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)' ,'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
#dataset = pandas.read_csv(url, sep=',',names=names)

url = "prsa.csv"
names = ['year','month','day','hour','DEWP','TEMP','PRES','Iws','pm2.5','Predicted pm2.5']
dataset = pandas.read_csv(url,names=names)
#
#
## shape
#print(dataset.shape)
#print ("\n\n\n")
#
#print(dataset.columns)
#
## head
#print(dataset.head(2))
#print ("\n\n\n")
##
#print(dataset.tail(2))
#print ("\n\n\n")
##
### descriptions
#print(dataset.describe())
#print ("\n\n\n")
##
### class distribution
#print(dataset.groupby('result').size())
#print ("\n\n\n")
##
## box and whisker plots
#dataset.plot(kind='box', subplots=True)
#plt.show()
##
### histograms
#dataset.hist()
#plt.show()
##
### scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

## Spot Check Algorithms
#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
##models.append(('SVM', SVC()))
## evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#    
## Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()
#
## Make predictions on validation dataset
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)
#predictions = knn.predict(X_validation)
#
#print("accuracy =")
#accuracy = knn.score(X_train, Y_train)
#print(accuracy)
#print("\n\n")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#print("\n\n")
example_measures = np.array([2050,1,2,21,-7,-5,1090,9.17,123])
example_measures=example_measures.reshape(1 ,-1)
#prediction = knn.predict(example_measures)
#print(prediction)
#lr = LogisticRegression()
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_validation)
##print(prediction)
#prediction = lr.predict(example_measures)
#print(prediction)

reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
reg.intercept_
print('variance_score: %.2f' % reg.score(X_validation,Y_validation))
predic=reg.predict(X_validation)
predic = reg.predict(example_measures)
print(predic)