# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

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
data2=pd.read_csv("dataset.csv")
temp=data2.as_matrix()
temp=temp.astype(int)
sub=temp[:,12:17]-temp[:,18:23];
print(sub)
subtraction=np.zeros(len(sub))
for index in range(1,len(sub)):
    subtraction[index]=np.sum(sub[index])
print(len(subtraction))
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

tar=pd.DataFrame(targets[1:len(targets)-2])
tar.drop(tar.index[[0]])
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
#plt.scatter(reduced_data['Dimension 1'][ii],reduced_data['Dimension 2'][ii],color='red')
#plt.scatter(reduced_data['Dimension 1'][jj],reduced_data['Dimension 2'][jj],color='blue')



log_data = math.log(reduced_data)
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
plt.show()