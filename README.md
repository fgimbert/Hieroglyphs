# Hieroglyphs

Personal project where I play with machine learning on Egyptian Hieroglyphs.

DBSCAN / Old implementation of a DBSCAN (clustering) algorithm to localize hieroglyphs in a picture. Not working anymore.

Currently on-going : implementation of siamese neural networks similar to face recognition (FaceNet). Using the dataset
available on https://github.com/morrisfranken/glyphreader

First try with siamese neural networks and triplet loss done in hieroRecognition.py. It's working ! 

True Hieroglyph :  S29 // Predicted :  S29 dist :  0.5425345318390896

True Hieroglyph :  D21 // Predicted :  D21 dist :  0.5125530245953122

True Hieroglyph :  G17 // Predicted :  G21 dist :  0.41295210841828955

![alt text](screenshots/results2.png "Left : Input Hieroglyph // Right : Predicted class")


TO DO LIST : 
- check all the variables used and remove the useless
- implement a way to visualize the results with pictures and not the labels - done
- improve the neural network - in work
- improve the database used for train and test
- use the model with transfer learning already implemented (model_online) convert to RGB
- implement a contrastive loss
- write an explanation of the model 

Next step : hieroglyph localization inside picture ? 
