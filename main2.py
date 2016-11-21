### Load the Libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

### Load the dataset, remove repeated values.
data = pd.read_csv('dataset.csv', header=1)
data.shape
data.head()
data = data.drop(["ID"],axis=1)
data_clean = data.drop_duplicates()
#####   Split DataFrame in Train and Test Set
train=data_clean.sample(frac=0.8,random_state=200)
test=data_clean.drop(train.index)
print(len(train))
print(data_clean.head())
X_train=train.values[:,:23]
X_test=test.values[:,:23]
y_train=train.values[:,23]
y_test=test.values[:,23]
print(len(y_test))

print(data_clean.columns)
data_clean.loc[:,['LIMIT_BAL','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].describe()
######## Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
score=clf.score(X_train, y_train)
print ('The accuracy for training Decision Tree is : %.2f ' % (score))

print('The accuracy for testing Decision Tree is : %.2f ' % clf.score(X_test,y_test))
######## K Nearest Neighbor Classifier
clf2 = KNeighborsClassifier(n_neighbors=5)
clf2.fit(X_train, y_train)
score2=clf2.score(X_train, y_train)
print ('The accuracy for training KNN is : %.2f ' % (score2))

print('The accuracy for testing KNN is : %.2f ' % clf2.score(X_test,y_test))
######## Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf3=GaussianNB()
clf3.fit(X_train, y_train)
score3=clf3.score(X_train, y_train)
print ('The accuracy for training Gaussian NB is : %.2f ' % (score3))

print('The accuracy for testing Gaussian NB is : %.2f ' % clf3.score(X_test,y_test))
######## Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(random_state=0)
clf4.fit(X_train, y_train)
score4=clf4.score(X_train, y_train)
print ('The accuracy for training Random Forest is : %.2f ' % (score4))

print('The accuracy for testing Random Forest is : %.2f ' % clf4.score(X_test,y_test))
######## Ada Boost Classifier
from sklearn.ensemble import AdaBoostClassifier
clf5 = AdaBoostClassifier(random_state=0)
clf5.fit(X_train, y_train)
score5=clf5.score(X_train, y_train)
print ('The accuracy for training Ada Boost is : %.2f ' % (score5))

print('The accuracy for testing Ada Boost is : %.2f ' % clf5.score(X_test,y_test))

######## Gradiant Boost Classifier
from sklearn.ensemble import GradientBoostingClassifier

clf6 = GradientBoostingClassifier(random_state=0)
clf6.fit(X_train, y_train)
score6=clf6.score(X_train, y_train)
print ('The accuracy for training Gradiant Boosting is : %.2f ' % (score6))

print('The accuracy for testing Gradiant Boosting is : %.2f ' % clf6.score(X_test,y_test))
######## Decision Tree Gridsearch
from sklearn import grid_search
parameters = {'presort':('true','false'),'splitter':('random', 'best'),'criterion':('gini', 'entropy'),'min_samples_split':[20,30], 'max_depth':[5, 35]}

grid_obj = grid_search.GridSearchCV(clf,parameters)

grid_obj.fit(X_train, y_train)
clf_o = grid_obj.best_estimator_
score7=clf_o.score(X_train, y_train)

print ('The accuracy for training Decision Tree (Optimised) is : %.2f ' % (score7))

print('The accuracy for testing Decision Tree (Optimised) is : %.2f ' % clf_o.score(X_test,y_test))
######## Random Forest Grid Search
#from sklearn import grid_search
parameters = {'criterion':('gini', 'entropy'),'min_samples_split':[20,30], 'max_depth':[5, 35],'n_estimators':[10,100]}

grid_obj2 = grid_search.GridSearchCV(clf4,parameters)

grid_obj2.fit(X_train, y_train)
clf_o2 = grid_obj2.best_estimator_

score8=clf_o2.score(X_train, y_train)

print ('The accuracy for training Random Forest (Optimised) is : %.2f ' % (score8))

print('The accuracy for testing Random Forest (Optimised) is : %.2f ' % clf_o2.score(X_test,y_test))
#### Store the classifier ( Rename clf_o to clf_2 to store Random Forest)
from sklearn.externals import joblib
joblib.dump(clf_o, 'clf_o.pkl')

print('Classifier Dumped in the clf_i.pkl file')