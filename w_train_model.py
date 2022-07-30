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
import time
import datetime
import csv
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *
from scipy.signal import *
from numpy import *

# Set the frame and step size
frame_size_seconds = 6
step_size_seconds = int(frame_size_seconds/2)
sampling_rate = 25

frame_size = frame_size_seconds * sampling_rate
step_size = step_size_seconds * sampling_rate

# -----------------------------------------------------------------------------------
#
#									
#
#
#
#								Train Model
#
#
#
#
#
#
# -----------------------------------------------------------------------------------


ts = time.time()
current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

print ""
print "---------------------------------------------------------"
print ""
print ""
print ""
print ""
print "Train Model for All"
print ""
print current_time
print ""
print ""
print ""
print ""
print "---------------------------------------------------------"
print ""

first_time_in_exclude_loop = 1
for p_counter in xrange(1, 22, 1):

	if p_counter==14:
		continue
		
	try:
		print "Loading: " + "../participants/" + str(p_counter) + "/datafiles/waccel_tc_ss_label.csv"
		L_T = genfromtxt("../participants/" + str(p_counter) + "/datafiles/waccel_tc_ss_label.csv", delimiter=',')
	except:
		error_participant_string = str(p_counter)
		print "** Error loading data for participant: " + str(p_counter)
		continue

	# Remove the relative timestamp
	L_T = L_T[:,1:]

	if first_time_in_exclude_loop==1:
		first_time_in_exclude_loop = 0
		Z_T = L_T
	else:
		Z_T = vstack((Z_T,L_T))

print ""
print "Shape of training data: " + str(Z_T.shape)
print ""
print str(Z_T)
print ""

# Number of inputs
number_of_inputs = Z_T.shape[1]-1

# -----------------------------------------------------------------------------------
#
#									Training
#
# -----------------------------------------------------------------------------------

print ""
print "---------------------------------------------------------"
print " Loading Features + Build Model"
print "---------------------------------------------------------"
print ""

pos_examples_counter = 0
neg_examples_counter = 0

# Calculate features for frame
for counter in xrange(0,len(Z_T),step_size):

	# Add up labels
	A_T = Z_T[counter:counter+frame_size, number_of_inputs]
	S_T = sum(A_T)

	if S_T>step_size:
		pos_examples_counter = pos_examples_counter + 1
		S_T = 1
	else:
		neg_examples_counter = neg_examples_counter + 1
		S_T = 0

	R_T = Z_T[counter:counter+frame_size, :number_of_inputs] 

	M_T = mean(R_T,axis=0)
	V_T = var(R_T,axis=0)
	SK_T = stats.skew(R_T,axis=0)
	K_T = stats.kurtosis(R_T,axis=0)
	RMS_T = sqrt(mean(R_T**2,axis=0))

	H_T = hstack((M_T,V_T))
	H_T = hstack((H_T,SK_T))
	H_T = hstack((H_T,K_T))
	H_T = hstack((H_T,RMS_T))

	# ----------------------------- Label -------------------------------------

	# Add label
	H_T = hstack((H_T,S_T))
	if counter==0:
		F_T = H_T
	else:
		F_T = vstack((F_T,H_T))
		# if S_T==1:
		# 	for p_counter in xrange(0,5,1):
		# 		F_T = vstack((F_T,H_T))

print ""
print "Positive Examples: " + str(pos_examples_counter)
print "Negative Examples: " + str(neg_examples_counter)
print ""


print ""
print "Print of F_T: " + str(F_T)
print ""

# Get features and labels
X_T = F_T[:,:number_of_inputs*5]
Y_T = F_T[:,number_of_inputs*5]

print ""
print "X_T: " + str(X_T)

print ""
print "Shape of X_T: " +str(X_T.shape)

print ""
print "Y_T: " + str(Y_T)

print ""
print "Shape of Y_T: " + str(Y_T.shape)

# Train classifier
#clf = ExtraTreesClassifier(n_estimators=100)
clf = RandomForestClassifier(n_estimators=185)
#clf = AdaBoostClassifier(n_estimators=185)
#clf = KNeighborsClassifier(n_neighbors=2)
#clf = svm.LinearSVC()
#clf = GaussianNB()
#clf = DecisionTreeClassifier()
#clf = LogisticRegression()

print ""
print "Training model..."
clf.fit(X_T,Y_T)

print ""
print "Saving model..."
joblib.dump(clf, '../model/wrist.pkl')


		