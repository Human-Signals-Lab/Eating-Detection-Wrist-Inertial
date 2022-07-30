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

import csv, re, datetime
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("pnumber")
args = parser.parse_args()
print "w_annotation.py"
print args.pnumber


# -------------------------------------------------------------------------------
#
# 							Load Activities
#
# -------------------------------------------------------------------------------

activities_time = []
activities_eatingflag = []

# Load annotated events into lists
with open('../participants/' + args.pnumber + '/datafiles/annotations-sorted.csv', 'rb') as csvinputfile:

	csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')

	for row in csvreader:
		activities_time.append(float(row[1]))
		activities_eatingflag.append(float(row[2]))

# -----------------------------------------------------------------------------------
#	get_activity_type_given_time
# -----------------------------------------------------------------------------------
def get_activity_type_given_time(query_time):

	# Iterate over activities and find out which one 'predicted' falls in
	last_activity_time = 0
	last_activity_eatingflag = 0

	for time_counter in xrange(0,len(activities_time)):

		current_activity_time = activities_time[time_counter]
		current_activity_eatingflag = activities_eatingflag[time_counter]

		if ((query_time>=last_activity_time) and (query_time<=current_activity_time)):
			break

		last_activity_time = current_activity_time
		last_activity_eatingflag = current_activity_eatingflag

	return last_activity_eatingflag


# -------------------------------------------------------------------------------
#
# 							Sort Chewing Annotations By Time
#
# -------------------------------------------------------------------------------

time_label_list = []

# Load annotated events into lists
with open('../participants/' + args.pnumber + '/datafiles/chewing-annotations.csv', 'rb') as csvinputfile:

	csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')
	
	# Skip header of annotation file
	csvreader.next()

	# The writer
	chewing_annotations_sorted_writer = csv.writer(open("../participants/" + args.pnumber + "/datafiles/chewing-annotations-sorted.csv", "wb"))

	for row in csvreader:
		
		# Get the data
		start_time_string = row[0]
		
		if row[4]=="Chewing":
			label = 1
		else:
			label = 0

		start_time_string_list = re.findall(r"[\d']+", start_time_string)
		start_time_seconds = int(start_time_string_list[0]) * 60 + int(start_time_string_list[1]) + (float(start_time_string_list[2])/10)

		time_label = []
		time_label.append(start_time_string);
		time_label.append(start_time_seconds);
		time_label.append(label);

		time_label_list.append(time_label)

	sorted_time_label_list = sorted(time_label_list, key=lambda x: x[1])

	for item in sorted_time_label_list:
		chewing_annotations_sorted_writer.writerow([item[0], item[1], item[2]])



# -------------------------------------------------------------------------------
#
# 				Assigning Labels by Eating vs. Non-Eating
#
# -------------------------------------------------------------------------------

with open("../participants/" + args.pnumber + "/datafiles/waccel_tc_ss_label.csv", 'wb') as csvoutputfile:

	csvwriter = csv.writer(csvoutputfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		
	# Open one
	with open('../participants/' + args.pnumber + "/datafiles/waccel_tc_ss.csv", 'rb') as csvinputfile:

		csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')
		
		event_list_index = 0
		label = 0

		next_event_secs = sorted_time_label_list[event_list_index][1]
		next_event_label = sorted_time_label_list[event_list_index][2]

		for row in csvreader:

			label = 0
			row_secs = float(row[0])

			if row_secs>(next_event_secs+3):
				
				next_event_label = 0
				while (next_event_label==0) and (event_list_index<len(sorted_time_label_list)-1):
					
					event_list_index = event_list_index + 1
					
					next_event_secs = sorted_time_label_list[event_list_index][1]
					next_event_label = sorted_time_label_list[event_list_index][2]
					
					if next_event_label==1:
						break

			elif row_secs>(next_event_secs-3) and next_event_label==1:
				label = get_activity_type_given_time(row_secs)
				
				if 0<label<4:
					label = 1 # eating
				else:
					label = 0 # not eating
				
			#print str(row_secs) + " - " + str(next_event_secs) + " - " + str(label)
			row.append(label)
			csvwriter.writerow(row)


# -------------------------------------------------------------------------------
#
# 					Assigning Labels by Activity Type
#
# -------------------------------------------------------------------------------

# with open("../participants/" + args.pnumber + "/datafiles/waccel_tc_ss_label.csv", 'wb') as csvoutputfile:

# 	csvwriter = csv.writer(csvoutputfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		
# 	# Open one
# 	with open('../participants/' + args.pnumber + "/datafiles/waccel_tc_ss.csv", 'rb') as csvinputfile:

# 		csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')
		
# 		label = 0

# 		for row in csvreader:

# 			row_secs = float(row[0])
# 			label = get_activity_type_given_time(row_secs)

# 			row.append(label)
# 			csvwriter.writerow(row)
