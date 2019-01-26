# Classification-Color-fundus-photos-transfer-learning-and-image-enhancement
We combined the image enhancement algorithm with transfer learning to achieve the classification of color fundus photo.


Firstly,we delete the meaningless photographs,crop the black area in the photographs use adjustbrightness.py to improve breghtness

then,use batchSplitGchannel.py and CLAHE.py.

Using flip.py and rotate.py to data augmentation.

The URL of Inception-V3 model parameters：https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

There are two files :1.imagenet_comp_graph_label_strings.txt 2.tensorflow_inception_graph.pb

Put them in model/ file，and creat tmp/bottleneck/ file Used to store the vectors for each image calculated by the Inception-v3 model

Directory Structure：

transfer-learning/

data/  
    fundus_photos/       
        0/   #Normal fundus photographs           
        1/   #Abnormal fundus photographs            
    tmp/      
        bottleneck/          
            ......              
model/   
    imagenet_comp_graph_label_strings.txt     
    tensorflow_inception_graph.pb     
train.py
train.py used for training the model

eval1.py used for predicting one prediction

eval2.py used for batch prediction and calculating accuracy, specificity, sensitivity

eval3.py used to save the probability of the model predicting each piece
roc_curve.py draw the ROC curve and calculate the AUC



