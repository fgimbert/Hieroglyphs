# Hieroglyphs

Personal project where I play with machine learning on Egyptian Hieroglyphs.

DBSCAN / Old implementation of a DBSCAN (clustering) algorithm to localize hieroglyphs in a picture. Not working anymore.

Currently on-going : implementation of siamese neural networks similar to face recognition (FaceNet). Using the dataset
available on https://github.com/morrisfranken/glyphreader
First try with siamese neural networks and triplet loss done in hieroRecognition.py. It's working ! 

True Hieroglyph :  N35 // Predicted :  N35 dist :  0.31444969855005006 
True Hieroglyph :  M17 // Predicted :  M1 dist :  0.7477856094253583
True Hieroglyph :  N35 // Predicted :  N35 dist :  0.5486129605025075
True Hieroglyph :  U1 // Predicted :  U1 dist :  0.7899228657901501
True Hieroglyph :  N35 // Predicted :  N35 dist :  0.519468589414221
True Hieroglyph :  G43 // Predicted :  G43 dist :  0.299870809479935
True Hieroglyph :  N35 // Predicted :  N35 dist :  0.46286286652256997
True Hieroglyph :  G17 // Predicted :  G17 dist :  0.47755075031076166
True Hieroglyph :  N35 // Predicted :  N35 dist :  0.48354205808368433 
True Hieroglyph :  G17 // Predicted :  G17 dist :  0.507103059385015 

TO DO LIST : 
- check all the variables used and remove the useless
- implement a way to visualize the results with pictures and not the labels.
- improve the database used for train and test
- use the model with transfer learning already implemented (model_online)
- implement a contrastive loss
- write an explanation of the model 

Next step : hieroglyph localization inside picture ? 
