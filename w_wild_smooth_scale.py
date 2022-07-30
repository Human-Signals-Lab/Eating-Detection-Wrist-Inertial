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

from scipy import *
from scipy.signal import *
from sklearn import preprocessing
from numpy import *
from argparse import ArgumentParser


# -----------------------------------------------------------------------------------
#	ema
# -----------------------------------------------------------------------------------
def ema(values, window):

    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()

    # Here, we will just allow the default since it is an EMA
    a = convolve(values, weights)[:len(values)]
    a[:window] = a[window]

    return a #again, as a numpy array.


parser = ArgumentParser()
parser.add_argument("pnumber")
args = parser.parse_args()
print "w_wild_smooth_scale.py"
print args.pnumber

filename = '../participants_wild/' + args.pnumber + '/wrist.csv'

# Read data from a text file
all_cols = genfromtxt( filename, comments='#', delimiter=",")

# Process only the data, ignore the time col and label
data_cols = all_cols[:, 1:all_cols.shape[1]-1]

# print str(data_cols)

data_cols_smoothened_0 = ema(data_cols[:,0], 10)
data_cols_smoothened_1 = ema(data_cols[:,1], 10)
data_cols_smoothened_2 = ema(data_cols[:,2], 10)

data_cols_smoothened = data_cols_smoothened_0
data_cols_smoothened = column_stack((data_cols_smoothened, data_cols_smoothened_1))
data_cols_smoothened = column_stack((data_cols_smoothened, data_cols_smoothened_2))

# Scale
# data_cols_smoothened_scaled = preprocessing.scale(data_cols_smoothened)

# MinMax Scale
# min_max_scaler = preprocessing.MinMaxScaler()
# data_cols_smoothened_scaled = min_max_scaler.fit_transform(data_cols_smoothened)

# Normalize
data_cols_smoothened_normalized = preprocessing.normalize(data_cols_smoothened, norm='l2')

# Add time and label
data_cols_smoothened_final = column_stack((all_cols[:,0],data_cols_smoothened_normalized))
data_cols_smoothened_final = column_stack((data_cols_smoothened_final,all_cols[:,all_cols.shape[1]-1]))

savetxt("../participants_wild/" + args.pnumber + "/wrist_ss.csv", data_cols_smoothened_final, delimiter=",")
