#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:11:55 2017

@author: markditsworth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def pca_results(good_data, pca):
    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions
    
    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    
    	# Create a bar plot visualization
    '''
    fig, ax = plt.subplots(figsize = (14,8))
    
    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar', legend=False);
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    
    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))
    '''
    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

# import raw data
dfData = pd.read_csv('secom.data.txt',delimiter=' ',header=None)
dfLabels = pd.read_csv('secom_labels.data.txt',delimiter=' ',header=None)
#print(dfLabels.head())

# get statistical data
metaData = dfData.describe()
#print(metaData.head())
s = dfData.shape[1]
# Remove sensors with missing data making up more than 20% of all data
print('Removing Empty Sensors...')
for sensor in metaData.columns:
    if metaData.loc['count',sensor] <= 0.8*dfData.shape[0]:
        dfData.drop([sensor],axis=1,inplace=True)

metaData = dfData.describe()
print('Reduced Dimensionality by: %d'%(s - dfData.shape[1]))
s = dfData.shape[1]
print('Number of Sensors Left: %d'%s)

# Fill missing data with averages
dfData.fillna(dfData.mean(),inplace=True)
metaData = dfData.describe()

# Feature Scale
from sklearn.preprocessing import MinMaxScaler
X = dfData.values
cols = dfData.columns
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

dfData_scaled = pd.DataFrame(X_scaled,columns=cols)
metaData = dfData_scaled.describe()
#print(metaData)

print('')
print('Removing Constant Sensors...')
# Remove sensors with std/mean below 0.3
for sensor in metaData.columns:
    if metaData.loc['mean',sensor] == 0:
        dfData_scaled.drop([sensor],axis=1,inplace=True)
    elif metaData.loc['std',sensor]/metaData.loc['mean',sensor] < 0.3:
        dfData_scaled.drop([sensor],axis=1,inplace=True)
        
# Store list of sensors that have made the cut
sensors = dfData_scaled.columns
print('Reduced Dimensionality by: %d'%(s - dfData_scaled.shape[1]))
s = dfData_scaled.shape[1]
print('Number of Sensors Left: %d'%s)

print('')
print('Removing Redundant Sensors...')

# get correlation coefficients dataframe
#import seaborn
#seaborn.heatmap(dfData_scaled.corr())
corrCoef = abs(dfData_scaled.corr())

# get pairs of features with high correlation coefficients
highCoefList = list(corrCoef[corrCoef > 0.9].stack().index)
# remove duplicate pairs and only select one of each pair
sensorsToDrop = []
for pair in highCoefList:
    if pair[0] < pair[1]:
        sensorsToDrop.append(pair[1])
sensorsToDrop = np.unique(sensorsToDrop)

# remove unnecessary sensors
dfData_scaled.drop(sensorsToDrop,axis=1,inplace=True)
print('Reduced Dimensionality by: %d'%(s - dfData_scaled.shape[1]))
print('Number of Sensors Left: %d'%dfData_scaled.shape[1])

#dfData_scaled = np.log(dfData_scaled)
# PCA
print(' ')
print('Performing PCA...')
from sklearn.decomposition import PCA
pca = PCA(n_components=dfData_scaled.shape[1],random_state=1).fit(dfData_scaled.values)
a = np.cumsum(pca.explained_variance_ratio_)
#plt.plot(a,'-o')
#plt.show()
i = 0
while a[i] < 0.9:
    i = i+1
print('Number of dimensions needed to capture 90%% of variance: %d'%i)
# PCA with reduced number of components
pca = PCA(n_components=i,random_state=1).fit(dfData_scaled.values)
pca_res = pca_results(dfData_scaled, pca)


# Construct Training Set and Test Set

# option 1: pass size equal to fail size
# Split data set into pass examples and fail examples
pass_index = dfLabels[dfLabels[0] == -1].index.values
fail_index = dfLabels[dfLabels[0] == 1].index.values
X_pass = dfData_scaled.loc[pass_index,:].values
X_fail = dfData_scaled.loc[fail_index,:].values
y_pass = dfLabels.loc[pass_index,0].values
y_fail = dfLabels.loc[fail_index,0].values
'''
f = plt.figure()
ax1 = f.add_subplot(221)
ax1.scatter(X_pass[:,5],X_pass[:,5],c='g')
ax1.scatter(X_fail[:,5],X_fail[:,7],c='r')
ax2 = f.add_subplot(222)
ax2.scatter(X_pass[:,8],X_pass[:,9],c='g')
ax2.scatter(X_fail[:,8],X_fail[:,9],c='r')
ax3 = f.add_subplot(223)
ax3.scatter(X_pass[:,3],X_pass[:,4],c='g')
ax3.scatter(X_fail[:,3],X_fail[:,4],c='r')
ax4 = f.add_subplot(224)
ax4.scatter(X_pass[:,5],X_pass[:,6],c='g')
ax4.scatter(X_fail[:,5],X_fail[:,6],c='r')
plt.show()
'''
# get random samples of X_pass and y_pass to match the fail size
fail_size = X_fail.shape[0]
pass_size = X_pass.shape[0]
idx = np.arange(pass_size)
np.random.shuffle(idx)
idx = idx[0:fail_size]
mask = np.ones(pass_size,dtype=bool)
mask[idx] = False

X_pass_selected = X_pass[idx,:]
X_pass_notselected = X_pass[mask,:]
y_pass_selected = y_pass[idx]
y_pass_notselected = y_pass[mask]

X = np.concatenate((X_pass_selected,X_fail))
y = np.concatenate((y_pass_selected,y_fail))

X_pca = pca.transform(X)
X_notselected = pca.transform(X_pass_notselected)

# option 2: class weights
'''
#   implemented in model initialization
X_pca = dfData_scaled.values
y = dfLabels.loc[:,0].values
'''
# option 3: SMOTE
'''
from imblearn.over_sampling import SMOTE
X = dfData_scaled.values
y = dfLabels.loc[:,0].values
X, y = SMOTE(ratio='minority',k_neighbors=10,kind='borderline2',random_state=0).fit_sample(X, y)
X_pca = pca.transform(X)
'''

# Split into train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, 
                                                    test_size = 0.25, 
                                                    random_state = random.randint(1,50))


print('')
print('Training Model...')
# Train Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef
clf = LogisticRegression(random_state=0,class_weight={1:14})
parameters = {'C':[0.01,0.02,0.05,0.1,0.2,0.4,0.5,0.7,0.8,1,1.2,1.5,2]}
scorer = make_scorer(matthews_corrcoef)
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
grid_fit = grid_obj.fit(X_train,y_train)
best_clf = grid_fit.best_estimator_

# Test the Model
y_pred = best_clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
matthews = matthews_corrcoef(y_test,y_pred)
print(' ')
print('Logistic Regression')
print('Accuracy: %.3f'%acc)
print('MCC:      %.3f'%matthews)

# Train SVM
from sklearn.svm import SVC
clf = SVC(random_state=0,class_weight={1:14})
parameters = {'kernel':['rbf','poly'],'degree':[3,4,5],'C':[0.01,0.05,0.1,0.2,0.5,0.7,1,1.5,2]}
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
grid_fit = grid_obj.fit(X_train,y_train)
best_clf = grid_fit.best_estimator_

# Test the Model
y_pred = best_clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
matthews = matthews_corrcoef(y_test,y_pred)
print(' ')
print('SVM')
print('Accuracy: %.3f'%acc)
print('MCC:      %.3f'%matthews)

# Train Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,class_weight={1:14})
parameters = {'max_depth':np.arange(10,100,10)}
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
grid_fit = grid_obj.fit(X_train,y_train)
best_clf = grid_fit.best_estimator_

# Test the Model
y_pred = best_clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
matthews = matthews_corrcoef(y_test,y_pred)
print(' ')
print('Decision Tree')
print('Accuracy: %.3f'%acc)
print('MCC:      %.3f'%matthews)


# Train Ensemble Learner
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=0)
parameters = {'n_estimators':np.arange(50,100,10),'learning_rate':[0.1,0.5,1,2,5]}
#scorer = make_scorer(matthews_corrcoef)
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
grid_fit = grid_obj.fit(X_train,y_train)
best_clf = grid_fit.best_estimator_

# Test the Model
y_pred = best_clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
matthews = matthews_corrcoef(y_test,y_pred)
print(' ')
print('AdaBoost')
print('Accuracy: %.3f'%acc)
print('MCC:      %.3f'%matthews)