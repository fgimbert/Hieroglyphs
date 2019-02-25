import os
import pandas as pd
import numpy as np
import random
#from PIL import Image
import numpy as np
from scipy.ndimage import imread
from generator import DataGenerator


path_files = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/"


def load_data(path=path_files):

    folders = next(os.walk(path))[1]
    labels = {}
    hieroglyphs = {}
    i = 0
    for folder in folders:
        # print(folder)
        for img_file in os.listdir(path+folder):
            # print(img_file)
            name, label = img_file.strip('.png').split("_")
            i+=1

            if label in labels.keys():
                labels[label].append(folder+"_"+name)
            else:
                labels[label] = [folder+"_"+name]

            hieroglyphs[folder+"_"+name] = [label]

    # Remove hireoglyph label with less than 5 examples
    for k, v in list(labels.items()):
        if len(v) <= 5:
            del labels[k]

    datahiero = pd.DataFrame.from_dict(hieroglyphs, orient='index')
    datahiero.columns = ["label"]
    datahiero = datahiero[datahiero.label != 'UNKNOWN']
    datahiero = datahiero.loc[datahiero['label'].isin(labels)]
    datahiero.reset_index(level=0, inplace=True)

    return datahiero, labels

def loadPictures(data):

    N_hieros = len(data)
    repertory, file = data['anchor'][0].split("_")
    label=str(data['label'][0])

    picture="/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/"+str(repertory)+"/"+str(file)+"_"+label+".png"

    #im = Image.open(picture)


    #img_x=im.size[0]
    #img_y=im.size[1]

    img_x=50
    img_y=75

    anchor, positive, negative = np.zeros((N_hieros,img_x*img_y)),np.zeros((N_hieros,img_x*img_y)),np.zeros((N_hieros,img_x*img_y))
    labels_true = []
    labels_wrong= []


    for index, row in data.iterrows():

        repertory, file = row['anchor'].split("_")
        label = row['label']
        picture = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/" + str(repertory) + "/" + str(
            file) + "_" + str(label) + ".png"
        labels_true.append(label)
        anchor[index]=imread(picture, flatten=True).reshape(1,img_x*img_y)

        repertory, file = row['positive'].split("_")
        picture = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/" + str(repertory) + "/" + str(
            file) + "_" + str(label) + ".png"
        positive[index] = imread(picture, flatten=True).reshape(1, img_x * img_y)

        repertory, file = row['negative'].split("_")
        label = row['neg_label']
        picture = "/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/" + str(repertory) + "/" + str(
            file) + "_" + str(label) + ".png"
        labels_wrong.append(label)
        negative[index] = imread(picture, flatten=True).reshape(1, img_x * img_y)

    return [anchor,positive,negative],labels_true,labels_wrong


datahiero, labels = load_data()
print(datahiero.shape)
print(len(labels))

datahiero.to_csv('datahiero.csv', index=False)

df = pd.read_csv('datahiero.csv', sep=',', header=None)
numpy = df.values

print(numpy)