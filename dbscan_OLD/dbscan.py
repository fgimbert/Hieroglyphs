# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:53:47 2016

@author: floriangimbert
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 12:34:50 2016

@author: floriangimbert
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

hiero = ndimage.imread("test_hiero.png")



hiero2d = np.zeros((hiero.shape[0],hiero.shape[1]))

Height = hiero2d.shape[0]
Width = hiero2d.shape[1]
 
Tot=0

      
for i in range(Height):
    for j in range(Width):
        if hiero[i,j,1]!=255:
            hiero2d[i,j]=1
            Tot+=1



hiero_p = np.zeros((Tot,2))

idx=0        
for i in range(Height):
    for j in range(Width):
        if hiero2d[i,j]==1:
            hiero_p[idx,0]=i
            hiero_p[idx,1]=j
            idx+=1
    
    


class DBSCAN(object):

    def __init__(self, Data, Picture):
        """ initialisation // probably useless here .

    Parameters
    ------------
    Data : into pixels

    """
        self.hiero_pic = Picture
        self.pixels = Data
        
                
    
    def scanCluster(self, eps, MinPts):
        """Determine if a  datapoint is noise or create/expand a new cluster 
        if it is a core point

        Parameters
        ------------
        eps : size of a circle around a data point
        
        MinPts : Density minimum in the circle of size eps to not be noise

        Returns
        -----------
        cluster : clusters labels 
        """
        
        
        C = 1
        
        self.cluster=np.ones((self.pixels.shape[0],3))
        
        
        
        
        
        
        for idx in range(self.pixels.shape[0]):
            if self.cluster[idx,2]== 1:
                NeighborPts = self.regionQuery(self.pixels[idx,0],self.pixels[idx,1],eps)
                if (len(NeighborPts)<MinPts):
                    #print('Noise')
                    self.cluster[idx,2]=-1
                else:
                    C+=1
                    self.expandCluster(idx,NeighborPts,C,eps,MinPts)
                                                                                                     
        return self.hiero_pic,C-1               

    

    def expandCluster(self, idx, NeighborPts, C, eps, MinPts):
        """Expand a cluster from a core datapoint by checking 
        all the neighbor points. 

        Parameters
        ------------
        idx : index from the first core point 
        
        NeighborPts : Density  in the circle of size eps 
        
        C : current cluster


        """
        
        self.cluster[idx,2]=C
                       
        for pts in NeighborPts:  
            print(pts[0],pts[1])
            if self.hiero_pic[int(pts[0]),int(pts[1])] == 1:               
                self.hiero_pic[pts[0],pts[1]]=C
                NeighborPts_up = self.regionQuery(pts[0],pts[1], eps)
                if len(NeighborPts_up) >= MinPts:
                    print(len(NeighborPts_up))
                    #NeighborPts.append(NeighborPts_up)

                    NeighborPts+=  NeighborPts_up
                    print(len(NeighborPts))
      
#        for x in NeighborPts:           
#           if self.cluster[x]== 0:               
#                self.cluster[x]=C
#                NeighborPts_up = self.regionQuery(self.pixels[x,0],self.pixels[x,1], eps)
#                if len(NeighborPts_up) >= MinPts:
#                    NeighborPts+=  NeighborPts_up
#               

               

    def regionQuery(self,iP,jP, eps):
        """Calculate all the neighbor points of a point. 

        Parameters
        ------------
        iP : coordinate of the center 
        
        jP : coordinate of the center 
        
        eps : size
        
        Returns
        -----------
        N :  neighbors 

        """
        
        Position=[]
        temp_p=[]
        
        for x in range (2*eps):
            for y in range (2*eps):
                if iP-eps+x<0 or jP-eps+y<0:
                    continue
                if (iP-eps+x)>=Height or (jP-eps+y)>=Width:
                    continue
                if self.hiero_pic[iP-eps+x,jP-eps+y]==1:
                    temp_p=np.vstack([iP-eps+x,jP-eps+y])
                    Position.append([iP-eps+x,jP-eps+y])
                    #print(Position)
            
                    
#        N = []
#        for idx in range(self.pixels.shape[0]):
#            if (np.square(self.pixels[idx,0]-iP)+np.square(self.pixels[idx,1]-jP))<np.square(eps):
#                N+=[idx]
        return Position
                
    def drawBox(self, Nc):
            """Draw box around one cluster 
    
            Parameters
            ------------
           
            
            Returns
            -----------
            N :  neighbors 
    
            """
            
#            self.hiero_pic[pts[0],pts[1]]
            Np_c=[]
            
            for idx in range(2,Nc+2):
                temp=[]
                temp_f=[]
                for i in range(Height):
                    for j in range(Width):
                        if self.hiero_pic[i,j]==idx:
                            temp.append([i,j,idx-1])
                        
                temp_f=np.vstack(temp)
                Np_c.append(temp_f)
                
#                
#            for i in range(Height):
#                for j in range(Width):
#                    if self.hiero_pic[i,j]>1:
#                        self.cluster[idx,0]=i
#                        self.cluster[idx,1]=j
#                        self.cluster[idx,2]=self.hiero_pic[i,j]                       
#                        idx+=1
#            
#            Np_c=[]
#            
#            
#            
#            for idx in range(2,Nc+2):
#                temp=[]
#                temp_f=[]
#                for i in range(self.cluster.shape[0]):
#                    if self.cluster[i,2]==idx:
#                        temp.append(self.pixels[i,:])
#                        
#                temp_f=np.vstack(temp)
#                Np_c.append(temp_f)
#                
         
                
            return Np_c                   

            
                           

nn = DBSCAN(hiero_p,hiero2d)
y_db, Nc = nn.scanCluster(2,3)


print(y_db[45,48])
#hiero_c=hiero2d
#
#for i in range(hiero2d.shape[0]):
#    for j in range(hiero2d.shape[1]):
#        hiero_c[i,j]=255
# 
#for i in range(y_db.shape[0]):
#    hiero_c[hiero_p[i,0],hiero_p[i,1]]=int(y_db[i])*50


fig, ax = plt.subplots(ncols=1, nrows=1)

Nb_cluster = nn.drawBox(Nc)

print(Nb_cluster[1])
Cluster_size=np.zeros((Nc,5))

idx =0
for clust in Nb_cluster:
    xmin=min(clust[:,0])
    ymin=min(clust[:,1])
    xmax=max(clust[:,0])
    ymax=max(clust[:,1])
    Cluster_size[idx,0]=ymin
    Cluster_size[idx,1]=xmin
    Cluster_size[idx,2]=ymax-ymin
    Cluster_size[idx,3]=xmax-xmin
    Cluster_size[idx,4]=(ymax-ymin)*(xmax-xmin)
    idx+=1
    
MedianX=np.median(Cluster_size[:,2])
MedianY=np.median(Cluster_size[:,3])
MedianArea=np.median(Cluster_size[:,4])

    

print(Nc)

height=hiero2d.shape[0]/4


for clust in Nb_cluster:   
    xmin=min(clust[:,0])
    ymin=min(clust[:,1])
    xmax=max(clust[:,0])
    ymax=max(clust[:,1])  
    
    print(xmin, xmax, ymin, ymax)
    print(len(Nb_cluster))
    
    if ymax-ymin <2:
        print('remove')
#        Nb_cluster.remove(clust.all())
        continue
#    if xmax-xmin >= height:
#        print('remove')
# #       Nb_cluster.remove(clust.all())
#        continue
    
    if (ymax-ymin)*(xmax-xmin)<=5:
        print('remove')
#        Nb_cluster.remove(clust.all())
        continue
    
    rect = mpatches.Rectangle((ymin, xmin), ymax-ymin, xmax-xmin,fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

ax.imshow(hiero)

plt.savefig('hiero_dbscan_box.png', dpi=200)
plt.show()


         
print("finish")




