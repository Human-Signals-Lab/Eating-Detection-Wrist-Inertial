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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *
from scipy.signal import *
from numpy import *

participant_time_offset_list = [18,14,39,63.5,91,39,15,10,29,47,90,28,35,21,12,11,24,14,12,-18,14]

# -----------------------------------------------------------------------------------
#
#								Parameters
#
# -----------------------------------------------------------------------------------

param_sweep_file = csv.writer(open("../results/w_lopo_slidewin_param_sweep.csv", "wb",0))
param_sweep_file.writerow(["parameter", "a", "p", "r", "f", "e"])


# -----------------------------------------------------------------------------------
#
#								Parameter Sweep Loop
#
# -----------------------------------------------------------------------------------

for parameter in xrange(12, 250, 12):

	# Set the frame and step size
	frame_size = parameter
	step_size = int(parameter/2)

	error_participant_list_train = []
	error_participant_list_eval = []

	participant_eval_list = []

	total_tn = 0
	total_tp = 0
	total_fn = 0
	total_fp = 0

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

	results_per_participant = csv.writer(open("../results/w_lopo_slidewin_results_by_participant.csv", "wb",0))
	results_per_participant.writerow(["Participant", "Accuracy", "Precision", "Recall"])

	for active_participant_counter in xrange(1, 22, 1):

		if active_participant_counter==14:
			continue

		ts = time.time()
		current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

		print ""
		print "---------------------------------------------------------"
		print ""
		print ""
		print ""
		print ""
		print "Train Model for Participant: " + str(active_participant_counter)
		print ""
		print current_time
		print ""
		print ""
		print ""
		print ""
		print "---------------------------------------------------------"
		print ""

		first_time_in_exclude_loop = 1
		for exclude_active_participant_counter in xrange(1, 22, 1):

			if (exclude_active_participant_counter==active_participant_counter) or (exclude_active_participant_counter==14):
				print "Skip loading " + "../participants/" + str(exclude_active_participant_counter) + "/datafiles/waccel_tc_ss_label.csv"
				continue

			try:
				print "Loading: " + "../participants/" + str(exclude_active_participant_counter) + "/datafiles/waccel_tc_ss_label.csv"
				L_T = genfromtxt("../participants/" + str(exclude_active_participant_counter) + "/datafiles/waccel_tc_ss_label.csv", delimiter=',')
			except:
				error_participant_string = str(exclude_active_participant_counter)
				if error_participant_string not in error_participant_list_train:
					error_participant_list_train.append(error_participant_string)
				continue

			# Remove the relative timestamp
			L_T = L_T[:,1:]
			# L_T_Label = L_T[:,L_T.shape[1]-1]
			# L_T = L_T[:,:6]
			# L_T = column_stack((L_T, L_T_Label))

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
		print " Loading Features + Build Model for Participant: " + str(active_participant_counter)
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

		#Randomize rows
		# print ""
		# print "Randomizing frames..."
		# random.seed()
		# random.shuffle(All)

		# All_Random = zeros((len(All), ((number_of_inputs-1)*5) + 2))
		# for counter in xrange(0,len(All)):
		# 	sampled_row_number = random.randint(0,len(All)-1)
		# 	All_Random[counter,:] = All[sampled_row_number,:]

		# All = All_Random

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
		#clf = KNeighborsClassifier(n_neighbors=3)
		#clf = svm.LinearSVC()
		#clf = GaussianNB()
		#clf = DecisionTreeClassifier()
		#clf = LogisticRegression()
		#clf = SVC()

		clf.fit(X_T,Y_T)


		# -----------------------------------------------------------------------------------
		#
		#									
		#
		#
		#
		#								Evaluate Model
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
		print "Evaluate Model for Participant: " + str(active_participant_counter)
		print ""
		print current_time
		print ""
		print ""
		print ""
		print "---------------------------------------------------------"
		print ""

		try:
			L_E = genfromtxt("../participants/" + str(active_participant_counter) + "/datafiles/waccel_tc_ss_label.csv", delimiter=',')
		except:
			error_participant_eval_string = str(active_participant_counter)
			if error_participant_eval_string not in error_participant_list_eval:
				error_participant_list_eval.append(error_participant_eval_string)
			continue

		# Remove the relative timestamp
		L_E = L_E[:,1:]
		# L_E_Label = L_E[:,L_E.shape[1]-1]
		# L_E = L_E[:,:6]
		# L_E = column_stack((L_E, L_E_Label))

		# -----------------------------------------------------------------------------------
		#
		#								Save Ground Truth
		#
		# -----------------------------------------------------------------------------------

		pos_examples_counter = 0
		neg_examples_counter = 0

		# Save eating ground truth
		eating_gt = []
		for counter in xrange(0,len(L_E),step_size):

			# Add up labels
			A_E = L_E[counter:counter+frame_size, number_of_inputs]
			S_E = sum(A_E)

			if S_E>step_size:
				pos_examples_counter = pos_examples_counter + 1
				S_E = 1
			else:
				neg_examples_counter = neg_examples_counter + 1
				S_E = 0

			eating_gt.append(S_E)

		print ""
		print "Positive Examples: " + str(pos_examples_counter)
		print "Negative Examples: " + str(neg_examples_counter)
		print ""

		# -----------------------------------------------------------------------------------
		#
		#									Prediction
		#
		# -----------------------------------------------------------------------------------

		Z_E = L_E

		print ""
		print "Shape of Z_E: " + str(Z_E.shape)
		print ""
		print str(Z_E)
		print ""
		
		# Calculate features for frame
		for counter in xrange(0,len(Z_E),step_size):

			R_E = Z_E[counter:counter+frame_size,:number_of_inputs] # x y z

			M_E = mean(R_E,axis=0)
			V_E = var(R_E,axis=0)
			SK_E = stats.skew(R_E,axis=0)
			K_E = stats.kurtosis(R_E,axis=0)
			RMS_E = sqrt(mean(R_E**2,axis=0))

			H_E = hstack((M_E,V_E))
			H_E = hstack((H_E,SK_E))
			H_E = hstack((H_E,K_E))
			H_E = hstack((H_E,RMS_E))

			if counter==0:
				T_E = H_E
			else:
				T_E = vstack((T_E,H_E))

		print ""
		print "Shape of T_E: " + str(T_E.shape)


		# Predict clusters
		predicted = clf.predict(T_E)

		# Add ground truth next to predicted array for evaluation
		predicted = column_stack((predicted, eating_gt))

		print ""
		print "Shape of Predicted: " + str(predicted.shape)
		print ""


		# --------------- Ground Truth - Get times for all activities --------------

		activities_time = []
		activities_eatingflag = []

		# Load annotated events into lists
		with open('../participants/' + str(active_participant_counter) + '/datafiles/annotations-sorted.csv', 'rb') as csvinputfile:

			csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')

			print ""
			for row in csvreader:
				activities_time.append(float(row[1]))
				activities_eatingflag.append(float(row[2]))
				print "GT Activity Time/Label: " + str(row[1]) + " " + str(row[2])


		# -----------------------------------------------------------------------------------
		#
		#									Evaluation
		#
		# -----------------------------------------------------------------------------------


		output_file = csv.writer(open("../results/w_lopo_slidewin_" + str(active_participant_counter) + "_results.csv", "wb",0))
		output_file.writerow(["gt", "p"])
		
		tn = 0
		tp = 0
		fn = 0
		fp = 0

		for counter in xrange(0,len(predicted)):

			# Ground truth is predicted[counter,1], predictions are predicted[counter,0]

			if predicted[counter,1]==1 and predicted[counter,0]==1:
				tp = tp + 1
			elif predicted[counter,1]==1 and predicted[counter,0]==0:
				fn = fn + 1
			elif predicted[counter,1]==0 and predicted[counter,0]==1:
				fp = fp + 1
			elif predicted[counter,1]==0 and predicted[counter,0]==0:
				tn = tn + 1

			# Write intake gesture predictions, predicted events and actual events
			output_file.writerow([str(predicted[counter,1]), str(predicted[counter,0])])


		print ""
		print "---------------------------------------------------------"
		print "Precision/Recall for Participant: " + str(active_participant_counter)
		print "---------------------------------------------------------"
		print ""
		print "TP: " + str(tp)
		print "TN: " + str(tn)
		print "FP: " + str(fp)
		print "FN: " + str(fn)

		try:

			# Update totals
			total_fn = total_fn + fn
			total_fp = total_fp + fp
			total_tn = total_tn + tn
			total_tp = total_tp + tp
			
			# Precision/recall measures
			accuracy = (tp+tn) / (tp+tn+fp+fn)
			precision = tp / (tp+fp)
			recall = tp / (tp+fn)

			print ""
			print "Accuracy: " + str(accuracy)
			print "Precision: " + str(precision)
			print "Recall: " + str(recall)
			print ""
			print ""

			results_per_participant.writerow([str(active_participant_counter), str(accuracy), str(precision), str(recall)])

		except ZeroDivisionError:

			error_participant_list_eval.append(str(active_participant_counter))

			print ""
			print "---------------------------------------------------------"
			print "Error"
			print "---------------------------------------------------------"
			print ""
			print "Division by zero"
			print ""
			print ""

		# Save Z
		savetxt("../results/w_lopo_slidewin_z_e_" + str(active_participant_counter) + ".csv", Z_E, delimiter=",")

		# Indicate we considered this participant
		participant_eval_list.append(str(active_participant_counter))

	try:

		# Precision/recall measures
		total_accuracy = (total_tp+total_tn) / (total_tp+total_tn+total_fp+total_fn)
		total_precision = total_tp / (total_tp+total_fp)
		total_recall = total_tp / (total_tp+total_fn)
		total_fscore = (2*total_precision*total_recall) / (total_precision+total_recall)

		print ""
		print "---------------------------------------------------------"
		print "Total Precision/Recall"
		print "---------------------------------------------------------"
		print ""
		print "TP: " + str(total_tp)
		print "TN: " + str(total_tn)
		print "FP: " + str(total_fp)
		print "FN: " + str(total_fn)
		print ""
		print "Accuracy: " + str(total_accuracy)
		print "Precision: " + str(total_precision)
		print "Recall: " + str(total_recall)
		print ""
		print ""

		param_sweep_file.writerow([str(parameter), str(total_accuracy), str(total_precision), str(total_recall), str(total_fscore), str(len(error_participant_list_eval))])

	except ZeroDivisionError:

		print ""
		print "---------------------------------------------------------"
		print "Precision/Recall"
		print "---------------------------------------------------------"
		print ""
		print "Division by zero"
		print ""
		print ""


	print ""
	print "---------------------------------------------------------"
	print "Errors & Info"
	print "---------------------------------------------------------"
	print ""
	print "Error in training for participant: " + str(error_participant_list_train)
	print "Error in eval for participant: " + str(error_participant_list_eval)
	print ""



		