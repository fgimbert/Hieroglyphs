import os
import pandas


path="../../HieroDataset/Manual/Preprocessed/"


def loadData(folderPictures=path):
    folders=next(os.walk(folderPictures))[1]
    #print(folders)
    img_groups = {}
    img_list={}

    for folder in folders:
        for img_file in os.listdir(folderPictures+folder):
            #print(img_file)
            name, label = img_file.strip('.png').split("_")

        #print(name,label)
            if label in img_groups.keys():
                img_groups[label].append(folder+"_"+name)
            else:
                img_groups[label] = [folder+"_"+name]

            img_list[folder+"_"+name]=[label]




    dataHiero=pandas.DataFrame.from_dict(img_list,orient='index')
    dataHiero.columns = ["label"]
    dataHiero = dataHiero[dataHiero.label != 'UNKNOWN']
    dataHiero.reset_index(level=0, inplace=True)

    return dataHiero,img_groups


class load():
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

dataHiero,dictLabels=loadData()
print(dataHiero)

