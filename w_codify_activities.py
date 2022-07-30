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
print "w_codify_activities.py"
print args.pnumber

time_label_list = []

# -------------------------------------------------------------------------------
#
# 							Sort Annotations By Time
#
# -------------------------------------------------------------------------------

# Load annotated events into lists
with open('../participants/' + args.pnumber + '/datafiles/annotations.csv', 'rb') as csvinputfile:

	csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')
	
	# Skip header of annotation file
	csvreader.next()

	# The writer
	annotations_sorted_writer = csv.writer(open("../participants/" + args.pnumber + "/datafiles/annotations-sorted.csv", "wb"))

	for row in csvreader:
		
		# Get the data
		start_time_string = row[0]
		
		if "Eating FK" in row[4]:
			label = 1
		elif "Eating FS" in row[4]:
			label = 2
		elif "Eating Hand" in row[4]:
			label = 3
		elif "Movie" in row[4]:
			label = 4
		elif "Walking" in row[4]:
			label = 5
		elif "Talking" in row[4]:
			label = 6
		elif "Phone" in row[4]:
			label = 7
		elif "Comb" in row[4]:
			label = 8
		elif "Brush" in row[4]:
			label = 9
		elif "Waiting" in row[4]:
			label = 10
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
		annotations_sorted_writer.writerow([item[0], item[1], item[2]])

