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
print "w_timeconvert.py"
print args.pnumber

offset_list = [18,14,39,63.5,91,39,15,10,29,47,90,28,35,21,12,11,24,14,12,-18,14]

with open("../participants/" + args.pnumber + "/datafiles/waccel_tc.csv", 'wb') as csvoutputfile:

	csvwriter = csv.writer(csvoutputfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	with open('../participants/' + args.pnumber + '/datafiles/waccel.csv', 'rb') as csvinputfile:

		csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')

		file_line_counter = 0
		last_relative_time = 0
		relative_time = 0

		if int(args.pnumber)>len(offset_list):
			offset = 0
		else:
			offset = offset_list[int(args.pnumber)-1]

		for row in csvreader:

			file_line_counter += 1
			
			date_string = row[0] 
			date_string_list = re.findall(r"[\d']+", date_string)
			#print date_string_list
			milliseconds = float(date_string_list[6])/1000
			seconds = int(date_string_list[3]) * 3600 + int(date_string_list[4]) * 60 + int(date_string_list[5])
			row_time = seconds + milliseconds

			if file_line_counter==1:
				reference_time = row_time
				relative_time = 0 + offset
			else:
				delta = row_time - reference_time
				relative_time = delta + offset

				# if relative_time<last_relative_time:
				# 	relative_time = last_relative_time + 0.1

				#print delta

			last_relative_time = relative_time

			csvwriter.writerow([date_string,relative_time, row[1], row[2], row[3]])

				
