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

# -----------------------------------------------------------------------------------
#
#								Parameters
#
# -----------------------------------------------------------------------------------

param_sweep_file = csv.writer(open("../results/w_wild_lab_model_eval_wild7_param_sweep.csv", "wb",0))
param_sweep_file.writerow(["eps", "minpts", "a", "p", "r", "f", "e"])

# -----------------------------------------------------------------------------------
#
#								Parameter Sweep Loop
#
# -----------------------------------------------------------------------------------

for eps_parameter in xrange(10, 110, 10):
	for minpts_parameter in xrange(1, 6, 1):

		frame_size_seconds = 6
		step_size_seconds = int(frame_size_seconds/2)
		sampling_rate = 25

		# Set the frame and step size
		frame_size = frame_size_seconds * sampling_rate
		step_size = step_size_seconds * sampling_rate

		error_participant_list_eval = []
		participant_eval_list = []

		total_tn = 0
		total_tp = 0
		total_fn = 0
		total_fp = 0

		results_per_participant = csv.writer(open("../results/w_wild_lab_model_eval_wild7_results_by_participant.csv", "wb",0))
		results_per_participant.writerow(["Participant", "Accuracy", "Precision", "Recall", "F-Score"])

		for active_participant_counter in xrange(1, 8, 1):


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
			print "MinPts: " + str(minpts_parameter)
			print "EPS: " + str(eps_parameter)
			print ""
			print current_time
			print ""
			print ""
			print ""
			print "---------------------------------------------------------"
			print ""

			try:
				L_E = genfromtxt("../participants_wild/" + str(active_participant_counter) + "/wrist_ss.csv", delimiter=',')
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
			#									Prediction
			#
			# -----------------------------------------------------------------------------------

			Z_E = L_E

			# Number of inputs
			number_of_inputs = Z_E.shape[1]-1

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

			clf = joblib.load('../model/wrist.pkl')

			# Predict clusters
			predicted = clf.predict(T_E)

			print ""
			print "Shape of Predicted: " + str(predicted.shape)
			print ""
			

			# ---------------------------------- Find clusters of eating gestures ---------------------------

			print ""
			print "Findind Clusters of Predicted Eating Gestures..."

			predicted_clusters = []

			if sum(predicted)>0:

				# Fill in list with all predicted gestures
				predicted_cluster_array = []
				for counter in xrange(0,len(predicted)):
					if predicted[counter]==1:
						predicted_cluster_array.append(array([counter*step_size_seconds]))
					
				predicted_cluster_array = asarray(predicted_cluster_array)

				# print ""
				# print "Printing predicted_cluster_array: "
				# print str(predicted_cluster_array)
				# print ""
				# print "Printing predicted_cluster_array shape: "
				# print str(predicted_cluster_array.shape)

				# Do clustering
				dbscan = DBSCAN(min_samples=minpts_parameter, eps=eps_parameter)
				dbscan.fit(predicted_cluster_array)
				print ""
				print str(dbscan.labels_)

				# Find out mean of clusters
				last_cluster_label = 9999
				cluster_time_sum = 0
				cluster_element_counter = 0

				print ""
				for counter in xrange(0,len(predicted_cluster_array)):
					predicted_time = int(predicted_cluster_array[counter])
					predicted_cluster_label = dbscan.labels_[counter]

					# print str(predicted_time) + " - " + str(predicted_cluster_label)

					if ((predicted_cluster_label!=last_cluster_label) and (cluster_time_sum>0)):

						cluster_time_mean = int(cluster_time_sum / cluster_element_counter)
						print "Cluster " + str(last_cluster_label) + ": " + str(cluster_time_mean)

						if last_cluster_label>=0:
							predicted_clusters.append(cluster_time_mean)

						cluster_time_sum = 0
						cluster_element_counter = 0

					last_cluster_label = predicted_cluster_label
					cluster_time_sum = cluster_time_sum + predicted_time
					cluster_element_counter = cluster_element_counter + 1

				cluster_time_mean = int(cluster_time_sum / cluster_element_counter)
				print "Cluster " + str(last_cluster_label) + ": " + str(cluster_time_mean)


			# --------------- Ground Truth - Get times for all activities --------------

			activities_time = []
			activities_eatingflag = []

			last_label = 0

			print ""
			for counter in xrange(0,len(Z_E),1):

				current_label = Z_E[counter,number_of_inputs]

				if current_label!=last_label:
					
					activities_time.append(counter/sampling_rate)
					activities_eatingflag.append(current_label)

					print "GT Activity Time/Label: " + str(counter/sampling_rate) + " " + str(current_label)

					last_label = current_label

			# -----------------------------------------------------------------------------------
			#
			#									Evaluation
			#
			# -----------------------------------------------------------------------------------
			
			tn = 0
			tp = 0
			fn = 0
			fp = 0
			
			last_counter = 0

			eval_sliding_window_size_seconds = 3600
			eval_sliding_window_size = eval_sliding_window_size_seconds * sampling_rate
			eval_begin_time = 0

			print ""
			print "Evaluation"
			for counter in xrange(0,len(Z_E), eval_sliding_window_size):

				gt_eating = 0
				predicted_eating = 0

				eval_end_time = counter / sampling_rate

				# Iterate Eating GT
				for gt_counter in xrange(0,len(activities_time)):
					if eval_begin_time <= activities_time[gt_counter] <= eval_end_time:
						if activities_eatingflag[gt_counter]==1:
							gt_eating = 1
							break

				# Iterate Predicted GT
				for pred_counter in xrange(0,len(predicted_clusters)):
					if eval_begin_time <= predicted_clusters[pred_counter] <= eval_end_time:
						predicted_eating = 1
						break

				print ""
				print "Segment Begin/End: " + str(eval_begin_time) + " " + str(eval_end_time)
				print "GT Status: " + str(gt_eating)
				print "Predicted Status: " + str(predicted_eating)

				if gt_eating==1 and predicted_eating==1:
					tp = tp + 1
				elif gt_eating==1 and predicted_eating==0:
					fn = fn + 1
				elif gt_eating==0 and predicted_eating==1:
					fp = fp + 1
				elif gt_eating==0 and predicted_eating==0:
					tn = tn + 1

				eval_begin_time = eval_end_time
				last_counter = counter

			
			print ""
			print "Check remainder from loop..."

			if last_counter < len(Z_E):

				gt_eating = 0
				predicted_eating = 0
				
				eval_end_time = len(Z_E) / sampling_rate

				# Iterate Eating GT
				for gt_counter in xrange(0,len(activities_time)):
					if eval_begin_time <= activities_time[gt_counter] <= eval_end_time:
						if activities_eatingflag[gt_counter]==1:
							gt_eating = 1
							break

				# Iterate Predicted GT
				for pred_counter in xrange(0,len(predicted_clusters)):
					if eval_begin_time <= predicted_clusters[pred_counter] <= eval_end_time:
						predicted_eating = 1
						break

				print ""
				print "Segment Begin/End: " + str(eval_begin_time) + " " + str(eval_end_time)
				print "GT Status: " + str(gt_eating)
				print "Predicted Status: " + str(predicted_eating)

				if gt_eating==1 and predicted_eating==1:
					tp = tp + 1
				elif gt_eating==1 and predicted_eating==0:
					fn = fn + 1
				elif gt_eating==0 and predicted_eating==1:
					fp = fp + 1
				elif gt_eating==0 and predicted_eating==0:
					tn = tn + 1


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
				fscore = (2*precision*recall) / (precision+recall)

				print ""
				print "Accuracy: " + str(accuracy)
				print "Precision: " + str(precision)
				print "Recall: " + str(recall)
				print ""
				print ""

				results_per_participant.writerow([str(active_participant_counter), str(accuracy), str(precision), str(recall), str(fscore)])

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
			print "F-score: " + str(total_fscore)
			print ""
			print ""

			param_sweep_file.writerow([str(eps_parameter), str(minpts_parameter), str(total_accuracy), str(total_precision), str(total_recall), str(total_fscore), str(len(error_participant_list_eval))])

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
			print "Error in eval for participant: " + str(error_participant_list_eval)
			print ""



		



			