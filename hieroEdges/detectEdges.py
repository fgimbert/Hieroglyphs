import cv2 as cv
import numpy as np
import os
import pandas

path="/Users/floriangimbert/PycharmProjects/HieroDataset/Manual/Preprocessed/"


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

    return dataHiero,img_groups