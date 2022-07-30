#!/bin/sh

for i in 1 2 3 4 5 6 7 8
do
		echo ""
		echo ""
		echo "in-the-wild participant: $i"
		
		python w_wild_parse_db.py $i
		python w_wild_smooth_scale.py $i
done
