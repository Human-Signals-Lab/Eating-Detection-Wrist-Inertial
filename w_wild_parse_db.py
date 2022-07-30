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
from scipy import *
from numpy import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("participant_number")
args = parser.parse_args()
print "w_wild_parse_db.py"
print args.participant_number

master_list = []

participant_number = args.participant_number

try:
	csvreader = csv.DictReader(open("../participants_wild/" + str(participant_number) + "/raw.csv", 'rb'), delimiter=',', quotechar='"')

except IOError:
	print "IOError - opening file"

for row in csvreader:

	hour = float(row['hour'])
	minute = float(row['minute'])
	total_seconds = minute*60 + hour*3600

	act_label = row['act_label']
	if "Eating" in act_label:
		label = 1
	elif "Working" in act_label:
		label = 2
	elif "Drinking" in act_label:
		label = 3
	elif "Driving" in act_label:
		label = 4
	elif "Biking" in act_label:
		label = 5
	elif "Exercising" in act_label:
		label = 6
	elif "Shopping" in act_label:
		label = 7
	elif "Cooking" in act_label:
		label = 8
	elif "Cleaning" in act_label:
		label = 9
	elif "Gardening" in act_label:
		label = 10
	elif "Reading" in act_label:
		label = 11
	elif "Resting" in act_label:
		label = 12
	elif "TV" in act_label:
		label = 13
	elif "Family" in act_label:
		label = 14
	elif "Kids" in act_label:
		label = 15
	elif "Dogs" in act_label:
		label = 16
	elif "Chatting" in act_label:
		label = 17
	elif "Socializing" in act_label:
		label = 18
	elif "Meeting" in act_label:
		label = 19
	elif "Presenting" in act_label:
		label = 20
	elif "Presentation" in act_label:
		label = 21
	elif "Hygiene" in act_label:
		label = 22
	elif "Chores" in act_label:
		label = 23
	else:
		label = 0

	sensor = row['datatxt']
	sensor_triples = sensor.split(',')

	#print "Number of entries per row: " + str(len(sensor_triples))

	for sensor_triple in sensor_triples:						# [-48;-80;-992]

		if len(sensor_triple)==0:
			continue

		sensor_triple_no_brackets = sensor_triple[1:-1]			# -48;-80;-992
		sensor_values = sensor_triple_no_brackets.split(';')	# -48 -80 -992

		try:

			#print str(hour) + ":" + str(minute) + "," + str(sensor_values[0]) + "," + str(sensor_values[1]) + "," + str(sensor_values[2])
			#sensor_list = [float(sensor_values[0]), float(sensor_values[1]), float(sensor_values[2]), float(total_seconds), float(1)]
			sensor_list = [float(total_seconds), float(sensor_values[0]), float(sensor_values[1]), float(sensor_values[2]), label]

		except:

			#print "Error - " + str(sensor_values)
			continue

		master_list.append(sensor_list)

# Convert the list of sensor values to numpy array 
final_array = asarray(master_list)

savetxt("../participants_wild/" + str(participant_number) + "/wrist.csv", final_array, delimiter=",")