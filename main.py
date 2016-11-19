# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
#import renders as rs
import matplotlib.pyplot as plt
#import math

from sklearn.metrics import f1_score

df = pd.read_csv('dataset.csv', header=1)
df.shape
df = df.drop(["ID"],axis=1)
df_no_duplicates = df.drop_duplicates()

df_visualization = df_no_duplicates.rename(columns={"default payment next month":"default_payment"})

df_visualization.loc[(df_visualization.default_payment == 1), 'default_payment'] = 'YES'
df_visualization.loc[(df_visualization.default_payment == 0), 'default_payment'] = 'NO'

df_visualization.loc[(df_visualization.SEX == 1), 'SEX'] = 'MALE'
df_visualization.loc[(df_visualization.SEX == 2), 'SEX'] = 'FEMALE'

df_visualization.loc[(df_visualization.EDUCATION == 1), 'EDUCATION'] = 'GRADUATE_SCHOOL'
df_visualization.loc[(df_visualization.EDUCATION == 2), 'EDUCATION'] = 'UNIVERSITY'
df_visualization.loc[(df_visualization.EDUCATION == 3), 'EDUCATION'] = 'HIGH_SCHOOL'
df_visualization.loc[(df_visualization.EDUCATION == 4), 'EDUCATION'] = 'OTHERS'
df_visualization.loc[(df_visualization.EDUCATION == 5), 'EDUCATION'] = 'OTHERS'
df_visualization.loc[(df_visualization.EDUCATION == 6), 'EDUCATION'] = 'OTHERS'
df_visualization.loc[(df_visualization.EDUCATION == 0), 'EDUCATION'] = 'OTHERS'

df_visualization.loc[(df_visualization.MARRIAGE == 1), 'MARRIAGE'] = 'MARRIED'
df_visualization.loc[(df_visualization.MARRIAGE == 2), 'MARRIAGE'] = 'SINGLE'
df_visualization.loc[(df_visualization.MARRIAGE == 3), 'MARRIAGE'] = 'OTHERS'
df_visualization.loc[(df_visualization.MARRIAGE == 0), 'MARRIAGE'] = 'OTHERS'

df_visualization = df_visualization.copy(deep = True)

df_visualization["AGE_GROUP"] = 1



bins = [20,30,40,50,60]
df_visualization['AGE_GROUP'] = pd.cut(df_visualization['AGE'],bins)


file_train = df_no_duplicates.sample(frac=0.8,random_state=200)

file_test = df_no_duplicates.drop(file_train.index)

df_visualization.to_csv('final_dataset_dp.csv', encoding='utf-8')
file_train.to_csv('final_train_dp.csv', encoding='utf-8')
file_test.to_csv('final_test_dp.csv', encoding='utf-8')

print(file_train.dtypes)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
clf3 = KNeighborsClassifier(n_neighbors=5)
train_data = file_train.values
train_features = train_data[:,:23]
train_target = train_data[:,23]
clf3.fit(train_features, train_target)
rfc = rfc.fit(train_features, train_target)
score = rfc.score(train_features, train_target)

clf2=DecisionTreeClassifier(random_state=0)
clf2.fit(train_features, train_target)
score2=clf2.score(train_features, train_target)

print ('The accuracy for training 2 is : %.2f ' % (100*score2))
print ('The accuracy for training is : %.2f ' % (100*score))

test_data = file_test.values
test_features = test_data[:,:23]
test_target = test_data[:,23]

test_predicted = rfc.predict(test_features)

test_predicted2 = clf2.predict(test_features)
score2=clf2.score(test_features, test_target)
test_predicted3 = clf3.predict(test_features)
from sklearn.metrics import accuracy_score
print ('The accuracy for testing 3 is : %.2f ' % (100*accuracy_score(test_target, test_predicted3)))
print ('The accuracy for testing 2 is : %.2f ' % (100*accuracy_score(test_target, test_predicted2)))
print ('The accuracy for testing is : %.2f ' % (100*accuracy_score(test_target, test_predicted)))

Xtest_normalized = preprocessing.normalize(train_features, norm='l2')
#print(X_normalized)

parameters = {'presort':('true','false'),'splitter':('random', 'best'),'criterion':('gini', 'entropy'),'min_samples_split':[20,30], 'max_depth':[5, 35]}

f1_scorer = make_scorer(f1_score)
from sklearn import grid_search
# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(clf2,parameters)

grid_obj.fit(train_features, train_target)
clf_o = grid_obj.best_estimator_
print(train_features)
test_predicted4 = clf_o.predict(test_features)
print(grid_obj.best_params_)
print(test_features[0])
print ('The accuracy for testing 5 is : %.2f ' % (100*accuracy_score(test_target, test_predicted4)))
import pickle
s=pickle.dumps(clf_o)
from sklearn.externals import joblib
joblib.dump(clf_o, 'clf_o.pkl')
"""
from sklearn import grid_search
parameters = {'weights':('uniform','distance'),'algorithm':('auto','ball_tree','kd_tree','brute')}
clf_g = grid_search.GridSearchCV(clf3, parameters)
clf_g.fit(train_features, train_target)
clf4 = clf_g.best_estimator_
test_predicted4 = clf4.predict(test_features)

print ('The accuracy for testing 4 is : %.2f ' % (100*accuracy_score(test_target, test_predicted4)))"""
"""
# Show matplotlib plots inline (nicely formatted in the notebook)
#%matplotlib inline
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output


# Load the wholesale customers dataset
try:
    data = pd.read_csv("dataset.csv")
    data2_var=data
    #data=data.ix
    targets=data['Y']
    data = preprocess_features(data)
    print "Processed feature columns ({} total features):\n{}".format(len(data.columns), list(data.columns))
    #bill_p=data['X12','X13','X14','X15','X16','X17']
    #paid_p=data['X18','X19','X20','X21','X22','X23']
    #tot_p=bill_p-paid_p
    #print(tot_p.head())
    #print(targets)
    data2=pd.read_csv("dataset.csv")
    temp=data2.as_matrix()
    temp=temp.astype(int)
    sub=temp[:,12:17]-temp[:,18:23];
    print(sub)
    subtraction=np.zeros(len(sub))
    for index in range(1,len(sub)):
        subtraction[index]=np.sum(sub[index])
    print(len(subtraction))
    print(len(data))
    data.drop(['Y','test','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23'], axis = 1, inplace = True)
    #print(data.head())
    data['sub']=subtraction
    print(data.head())
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
#print(data.describe())
#pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
########data2=pd.read_csv("dataset.csv")
temp=data2.as_matrix()
temp=temp.astype(int)
sub=temp[:,12:17]-temp[:,18:23];
#print(sub)
subtraction=np.zeros(len(sub))
for index in range(1,len(sub)):
    subtraction[index]=np.sum(sub[index])
print(len(subtraction))##########
#print(np.add(temp[:,12:17]-temp[:,18:23]))
#bill_p1=pd.DataFrame([data2['X12'],data2['X13'],data2['X14'],data2['X15'],data2['X16'],data2['X17']])
#print(bill_p1)
#paid_p1=pd.DataFrame([data2['X18'],data2['X19'],data2['X20'],data2['X21'],data2['X22'],data2['X23']])

#print(data2.subtract[data2['X18']])

## Splitting the data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[1:], targets[1:], test_size=0.25, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

score = regressor.score(X_test,y_test)
print(score)
print(data['sub'])
X_normalized = preprocessing.normalize(data[1:], norm='l2')
#print(X_normalized)

pca = PCA(n_components=2).fit(X_normalized)
pca_reduced = pca.transform(X_normalized)

reduced_data = pd.DataFrame(pca_reduced, columns = ['Dimension 1', 'Dimension 2'])
#print(reduced_data.head())
#reduced_data=reduced_data.ix
#print(reduced_data['Dimension 1'])
#ii=find(targets>1)
#print(len(ii))
print(reduced_data.head())
reduced_data.drop(reduced_data.index[len(reduced_data)-1])

#tar=pd.DataFrame(targets[1:len(targets)-2])
#tar.drop(tar.index[[0]])
#print(tar)
#reduced_data['targets']=[targets,0]

reduced_data['Z']=targets[1:]
reduced_data.Z = reduced_data.Z.shift(-1)
print(reduced_data.head())
#print(reduced_data.head)
#print(pd.DataFrame(targets))
ii=reduced_data.Z=='1'
jj=reduced_data.Z=='0'
#print(ii)
#=reduced_data['Dimension 1','Z'>1]
#print(temp_dimension.head())
plt.scatter(reduced_data['Dimension 1'][ii],reduced_data['Dimension 2'][ii],color='red')
plt.scatter(reduced_data['Dimension 1'][jj],reduced_data['Dimension 2'][jj],color='blue')
plt.show()
"""
#pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
