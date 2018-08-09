import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import MultipleLocator, FormatStrFormatter



# load driving_log.csv
data_augmentation = False
if data_augmentation is True:
	work_path = './data_augmentation/'
else:
	work_path = './data/'
path = work_path + 'driving_log.csv' # original data
samples = []
angles = []
with open(path) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
    	angles.append(line[3])
    	samples.append(line)
    samples.pop(0) # remove the headline of description

angles = np.array(angles)
print(angles.shape)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.hist(angles,bins=1000,histtype='step')
ax1.locator_params(axis='x', nbins=5)
# ax1.ticklabel_format(style='sci')
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
plt.show()