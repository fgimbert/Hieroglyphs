import os
import pandas
import numpy as np
import random
#from PIL import Image

import numpy as np
from scipy.ndimage import imread

path="/Users/fgimbert/Documents/Dataset/Manual/Preprocessed/"


def loadData(folderPictures=path):

    folders=next(os.walk(folderPictures))[1]

    img_groups = {}
    img_list={}

    for folder in folders:
        for img_file in os.listdir(folderPictures+folder):
            name, label = img_file.strip('.png').split("_")

            # One image per class

            #if label not in img_groups.keys():
            #    img_groups[label] = [folder + "_" + name]


            # Multiple images per class

            if label in img_groups.keys():
                img_groups[label].append(folder+"_"+name)
            else:
                img_groups[label] = [folder+"_"+name]

            img_list[folder+"_"+name]=[label]


    # Remove class with only one hieroglyph


    for k,v in list(img_groups.items()):
        if len(v)==1: del img_groups[k]

    # Extract only N hieroglyph classes randomly

    nclass = 50

    list_of_class = random.sample(list(img_groups.keys()), nclass)
    #print(list_of_class)

    short_dico = {x: img_groups[x] for x in list_of_class if x in img_groups}





    dataHiero=pandas.DataFrame.from_dict(img_list,orient='index')
    dataHiero.columns = ["label"]
    dataHiero = dataHiero[dataHiero.label != 'UNKNOWN']

    dataHiero = dataHiero.loc[dataHiero['label'].isin(short_dico)]


    dataHiero.reset_index(level=0, inplace=True)

    return dataHiero,img_groups

def loadTriplets(dataset,labels):

    N_hieros=len(dataset)

    tripletHiero=[]

    for i in range(N_hieros):
        label=dataHiero['label'][i]
        hiero=dataHiero['index'][i]

        pos_hiero=labels.setdefault(label)
        positive=hiero

        while positive==hiero:
            positive=random.choice(pos_hiero)

        if positive==hiero: print('Positive Choice Error ! ')


        neg_label=label
        neg_labels=list(labels.keys())

        while neg_label==label or neg_label=='UNKNOWN':
            neg_label = random.choice(neg_labels)

        negative=random.choice(labels[neg_label])

        #if negative == hiero : print('Negative Choice Error ! ')

        tripletHiero.append([hiero,positive,negative,label,neg_label])
        dataTriplet =pandas.DataFrame(tripletHiero,columns=['anchor','positive','negative','label','neg_label'])



    return dataTriplet


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


#
#
# dataHiero,dictLabels=loadData(path)
#
# print(dataHiero.head())
# print(len(dataHiero)," hieroglyphs !")
# print(len(dictLabels.keys())," hieroglyphs different !")
#
#
# tripletData=loadTriplets(dataHiero,dictLabels)
#
# dataset,labels_true,labels_wrong =loadPictures(tripletData)
#
# train_data=dataset[0]
#
# print(train_data.shape)
