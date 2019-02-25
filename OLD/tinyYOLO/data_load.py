import numpy as np
import scipy.io
import random
import os

import csv
import itertools
import os
import pandas as pd

#folderLocation="/Users/floriangimbert/PycharmProjects/HieroDataset/Manual/Location/"
folderLocation="/Users/fgimbert/Documents/Dataset/Manual/Locations/"


datafile_rows=[]
with open('datafile.csv', 'w') as out_file:
    #writer.writerow(('picture', 'xmax', 'ymax', 'xmin', 'ymin','file'))
    for img_file in os.listdir(folderLocation):

        file=folderLocation+img_file
        with open(file, 'r') as in_file:

            for row in csv.reader(in_file):
                row=row[:5]
                row.append(img_file)
                row.append(folderLocation)
                row.append('hiero')
                #print(row)
                datafile_rows.append(row)

df = pd.DataFrame(datafile_rows, columns=['hiero', 'xmax', 'ymax', 'xmin', 'ymin','picture', 'folder', 'class'])

print(df.head(1))