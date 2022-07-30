#
# DISCLAIMER
#
# This script is copyright protected 2015 by
# Edison Thomaz, Irfan Essa, Gregory D. Abowd
#
# All software is provided free of charge and "as is", without
# warranty of any kind, express or implied. Under no circumstances
# and under no legal theory, whether in tort, contract, or otherwise,
# shall Edison Thomaz, Irfan Essa or Gregory D. Abowd  be liable to
# you or to any other person for any indirect, special, incidental,
# or consequential damages of any character including, without
# limitation, damages for loss of goodwill, work stoppage, computer
# failure or malfunction, or for any and all other damages or losses.
#
# If you do not agree with these terms, then you are advised to 
# not use this software.
#

from __future__ import division
import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import svm, neighbors, metrics, cross_validation
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, confusion_matrix
from scipy import *
from scipy.stats import *
from scipy.signal import *
from numpy import *

frame_size_seconds = 6
step_size_seconds = int(frame_size_seconds/2)
sampling_rate = 25

# Set the frame and step size
frame_size = frame_size_seconds * sampling_rate
step_size = step_size_seconds * sampling_rate

# Get training file for each participant
first_time_in_loop = 1
for target_participant_counter in xrange(1,22):

	if target_participant_counter==14:
		continue

	print ""
	print "---------------------------------------------------------"
	print " Participant: " + str(target_participant_counter)
	print "---------------------------------------------------------"
	print ""

	participant_file = '../participants/' + str(target_participant_counter) + '/datafiles/waccel_tc_ss_label.csv'

	print ""
	print "File: " + participant_file
	print ""

	D = genfromtxt(participant_file, delimiter=',')

	if first_time_in_loop==1:
		first_time_in_loop = 0
		Z = D
	else:
		Z = vstack((Z,D))

# Remove the relative timestamp
Z = Z[:,1:]

print ""
print "Shape of Z: " + str(Z.shape)

# Number of inputs
number_of_inputs = Z.shape[1] - 1

# -----------------------------------------------------------------------------------
#
#									Training
#
# -----------------------------------------------------------------------------------

# Calculate features for frame
for counter in xrange(0,len(Z),step_size):
	
	# Get the label
	L = Z[counter, number_of_inputs]

	# Get rows from which to calculate features
	R = Z[counter:counter+frame_size, :number_of_inputs]

	M = mean(R,axis=0)
	V = var(R,axis=0)
	SK = stats.skew(R,axis=0)
	K = stats.kurtosis(R,axis=0)
	RMS = sqrt(mean(R**2,axis=0))

	H = hstack((M,V))
	H = hstack((H,SK))
	H = hstack((H,K))
	H = hstack((H,RMS))

	H = hstack((H,L))
	if counter==0:
		All = H
	else:
		All = vstack((All,H))


# Get features and labels
X = All[:,:number_of_inputs*5]
Y = All[:,number_of_inputs*5]

print ""
print "X: " + str(X)

print ""
print "Shape of X: " +str(X.shape)

print ""
print "Y: " + str(Y)

print ""
print "Shape of Y: " + str(Y.shape)


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, random_state=42, test_size=0.33)

# Run classifier
classifier = RandomForestClassifier(n_estimators=185)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

#ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print (ct)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()







