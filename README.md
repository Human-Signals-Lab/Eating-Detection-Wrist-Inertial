# Eating-Detection-Wrist-Inertial

# Datasets

Data for the participants in the lab study can be downloaded at:

https://utexas.box.com/s/z1nckehnr6qdz24vq79sn7labz7ifug9

Data for the participants in the field studies can be downloaded at:

https://utexas.box.com/s/5rpzhfzp96iplq6nxngjbqxjpva2ynwp

# Directory Structure

The data processing pipeline relies on a specific directory structure. Here's how to create it:

1. Create a top level folder called "edwip" (for Eating Detection Wrist Inertial Pipeline).
2. Create a "bin" folder inside "edwip" and place all the python source files inside it.
3. Place the two participant data folders inside "edwip".
4. Create two empty folders inside "edwip": one called "model" and one called "results".

# Requirements

The scripts were tested on MacOS X 10.10.5 with the following packages installed:

- python 2.7.6
- numpy 1.8.0rc1
- scipy 0.13.0b1
- scikit-learn 0.15.2

# Laboratory Study Data

1. Run w_preprocessor_run.sh

The preprocessor will execute the following scripts:
	
- w_timeconvert.py, w_smooth_scale.py
- w_codify_activities.py
- w_annotation.py

The scripts will generate intermediary data files for each participant.

2. Run individual scripts:

- w_cm: 
	- Confusion matrix (see note below).
- w_lopo:
	- User independent evaluation over range of DBSCAN parameters.
- w_lopo_slidewin_param_sweep: 
	- User independent evaluation for different sliding window sizes.
- w_train_model: 
	- Trains a model and saves it in the ‘model’ folder.
	
Before running w_cm, make sure that label assignment is set to activity type and not eating vs. non-eating. 
Comment in/out the corresponding sections in w_annotation.py and re-run w_preprocessor.sh.

Data for participant 14 was not captured correctly and was removed from analysis and dataset.

Results will be saved in the ‘results’ folder.

# In-the-Wild Data (Wild-7 and Wild-Long)

- Wild-7: Participant data in folders 1 through 7
- Wild-Long: Participant data for one entire month (folder 8)

1. Run w_wild_preprocessor_run.sh

The preprocess will execute the following scripts:

- w_wild_parse_db.py
- w_wild_smooth_scale.py

The scripts will generate intermediary data files for each participant.

2. Run individual scripts:

- w_wild_lab_model_eval_timesegment_wild7.py:
	- Evaluating lab model on Wild-7 data for different time segments

- w_wild_lab_model_eval_timesegment_wildlong.py:
	- Evaluating lab model on Wild-long data for different time segments

- w_wild_lab_model_eval_wild7.py:
	- Evaluating lab model on Wild-7 data with time segment set to 1 hour over range of DBSCAN parameters

- w_wild_lab_model_eval_wildlong.py: 
	- Evaluating lab model on Wild-long data with time segment set to 1 hour over range of DBSCAN parameters

Results will be saved in the ‘results’ folder.


# Reference

If you use this dataset in your research, please reference our paper:

Edison Thomaz, Irfan Essa, and Gregory D. Abowd. 2015. A practical approach for recognizing eating moments with wrist-mounted inertial sensing. In Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '15). ACM, New York, NY, USA, 1029-1040. DOI=http://dx.doi.org/10.1145/2750858.2807545

@inproceedings{Thomaz:2015:PAR:2750858.2807545,
 author = {Thomaz, Edison and Essa, Irfan and Abowd, Gregory D.},
 title = {A Practical Approach for Recognizing Eating Moments with Wrist-mounted Inertial Sensing},
 booktitle = {Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
 series = {UbiComp '15},
 year = {2015},
 isbn = {978-1-4503-3574-4},
 location = {Osaka, Japan},
 pages = {1029--1040},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/2750858.2807545},
 doi = {10.1145/2750858.2807545},
 acmid = {2807545},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {activity recognition, automated dietary assessment, dietary intake, food journaling, inertial sensors},
} 


# Acknowledgements

The compilation of this work and resulting software package was possible thanks to the support of the Georgia Institute of Technology, the Intel Science and Technology Center for Pervasive Computing (ISTC-PC) and the Center of Excellence for Mobile Sensor Data-to-Knowledge (MD2K). 


# Contact

If you have any questions, please contact Edison Thomaz at http://www.ethomaz.com.
