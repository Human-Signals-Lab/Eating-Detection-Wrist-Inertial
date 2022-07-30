#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
do
		echo ""
		echo ""
		echo "participant: $i"
		
		python w_timeconvert.py $i
		python w_smooth_scale.py $i
		python w_codify_activities.py $i
		python w_annotation.py $i
done
