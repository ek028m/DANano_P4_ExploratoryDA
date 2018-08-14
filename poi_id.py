#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from numpy import mean

# %matplotlib inline

sys.path.append("../tools/")

from tester import dump_classifier_and_data, test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score


######## Task 1: Select what features you'll use. #################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'

features_list = ['poi',
                 'salary',
                 'total_payments',
                 'bonus',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',]
              
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Convert data into a pandas df    
df = pd.DataFrame.from_dict(data_dict, orient = 'index' , dtype = float)    

#==============================================================================
### Load the pkl file *** Investigate the pkl files ***
""" find how many POIs are in the dataset"""
i = 0
for key in data_dict:
    # value of poi field is type Boolean so don't use of string
    if data_dict[key]["poi"] == True:
        i = i + 1

print(i) # 18 POIs in the dataset

#==============================================================================
### Task 2: Remove outliers
df['poi'] # notice the TRAVEL AGENCY IN THE PARK poi entry

# Remove THE TRAVEL AGENCY IN THE PARK from the list of POIs since it's not 
# an actual person.  Data error entry.
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace = True)
df['poi']

df.plot.scatter(x = 'salary', y = 'bonus')

# outlier of a salary > 2.5 * 10e^7
# see who it belongs to
df['salary']
df['salary'].idxmax()

# remove this as it's the total of all the salaries of the employees in the Enron dataset
df.drop('TOTAL', inplace = True)
df.plot.scatter(x = 'salary', y = 'bonus')

# remove any outliers permanently
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# clean up the the data permanently
for ii in data_dict:
    for jj in data_dict[ii]:
        if data_dict[ii][jj] == 'NaN':
            data_dict[ii][jj] = 0

#==============================================================================

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Create new feature(s)

## create new features in the df
df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['to_messages']
# save to new features list
features_list.extend(['fraction_from_poi','fraction_to_poi'])

# len(my_feature_list) # count = 19 features

# remove Nan values from df
df.fillna(0, inplace=True)
df.round(decimals=1)
my_dataset = df.to_dict('index')

# k-best features: use for the SelectKBest parameter "k"
#num_features = 13
num_features = 15

# function for SelectKBest
def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    #format the scores
    formatted_scores = ['%.5f' % elem for elem in scores]
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    #create a dictionary out of the best features and their scores 
    table_dict = dict(zip(k_best_features, formatted_scores))
    print(table_dict)
    return k_best_features


best_features = get_k_best(my_dataset, features_list, num_features)
# print the feature-scores pairs in order of scores value
sorted_best_features = sorted(best_features.items(), key=operator.itemgetter(1))
print(sorted_best_features)   

my_feature_list = [target_label] + best_features.keys()
                     
# plot the features against each other
# plot non-poi
#plot1 = (df[df['poi'] == False].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', c='r', label='Non-POI'))
## plot of poi
#df[df['poi'] == True].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', c='b', label='POI', ax=plot1)


### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, my_feature_list)
y, X = targetFeatureSplit(data)
# define array space for the cross_validation
X = np.array(X)
y = np.array(y)

# properly scale the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#print(features)

#==============================================================================
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# gaussian NB
"""gaussian_clf = GaussianNB()

# decision tree
decisionTree_clf = DecisionTreeClassifier()

# k-means clustering
k_clf = KMeans(n_clusters=2, tol=0.001)

clf = decisionTree_clf
test_classifier(clf, my_dataset, my_feature_list)"""

#==============================================================================
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# cross-validation
# StratifiedShuffleSplit(n_splits=10, test_size=’default’, train_size=None, random_state=None)[source]¶
sss = StratifiedShuffleSplit(n_splits=10, test_size=.3, random_state=42)

def evaluate_model(grid, X, y, cv):
    nested_f1 = cross_val_score(grid, X=X, y=y, cv=cv, n_jobs=-1)
    print "Nested f1 score: {}".format(nested_f1.mean())
    
    grid.fit(X, y)
    
    accuracy = []
    precision = []
    recall = []
    f1 = []
     
    for train_index, test_index in cv.split(X, y): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        grid.best_estimator_.fit(X_train, y_train)
        pred = grid.best_estimator_.predict(X_test)
        
        accuracy.append(recall_score(y_test, pred))
        precision.append(f1_score(y_test, pred))        
        recall.append(recall_score(y_test, pred))
        f1.append(f1_score(y_test, pred))
    
    print "Accuracy: {} \n".format(np.mean(accuracy)), "Precision: {} \n".format(np.mean(precision)), \
        "Recall: {} \n".format(np.mean(recall)), "f1: {} \n".format(np.mean(f1))


""" Gaussian NB """
# define parameters for tuning the estimator
"""pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', GaussianNB())
        ])
# define gridsearchcv parameters
SCALER = [None]
#SELECTOR__K = [13]
SELECTOR__K = [15]
REDUCER__N_COMPONENTS = [6]

param_grid = {
        'scaler': SCALER,
        'selector__k': SELECTOR__K,
        'reducer__n_components': REDUCER__N_COMPONENTS,
        }

gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
# find all the available hyper-parameters of an estimator
#gnb_grid.get_params().keys()
evaluate_model(gnb_grid, X, y, sss)
test_classifier(gnb_grid.best_estimator_, my_dataset, my_feature_list)"""

""" Decision Tree """
# define pipeline parameters
pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', DecisionTreeClassifier())
        ])
# define parameters for tuning the estimator
SCALER = [StandardScaler()]
SELECTOR__K = [15]
REDUCER__N_COMPONENTS = [1]
CRITERION = ['gini']
SPLITTER = ['random']
MIN_SAMPLES_SPLIT = [6]
CLASS_WEIGHT = ['balanced']

param_grid = {
        'scaler': SCALER,
        'selector__k': SELECTOR__K,
        'reducer__n_components': REDUCER__N_COMPONENTS,
        'classifier__criterion': CRITERION,
        'classifier__splitter': SPLITTER,
        'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
        'classifier__class_weight': CLASS_WEIGHT
        }

dt_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
# find all the available hyper-parameters of an estimator
#dt_grid.get_params().keys()
evaluate_model(dt_grid, X, y, sss)
test_classifier(dt_grid.best_estimator_, my_dataset, my_feature_list)

""" K-Means Clustering """
# define parameters for tuning the estimator
"""pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', KMeans())
        ])
# define gridsearchcv parameters
SCALER = [None]
SELECTOR__K = [15]
REDUCER__N_COMPONENTS = [6]
N_CLUSTERS = [2]

param_grid = {
        'scaler': SCALER,
        'selector__k': SELECTOR__K,
        'reducer__n_components': REDUCER__N_COMPONENTS,
        'classifier__n_clusters': N_CLUSTERS
        }

k_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
# find all the available hyper-parameters of an estimator
#k_grid.get_params().keys()
evaluate_model(k_grid, X, y, sss)
test_classifier(k_grid.best_estimator_, my_dataset, my_feature_list)"""

# set the grid to most optimal
grid = dt_grid

#==============================================================================
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(grid, my_dataset, features_list)
