'''
1 Note the directory of runs
2 The input of the code is the path of the image.The format of the image is **0.JPG or **1.JPG, 
which is the name of the image with the label.

'''

import tensorflow as tf
import numpy as np
import os
# Model catalog
CHECKPOINT_DIR = './runs/1547178289/checkpoints'
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb' # Inception-v3 model parameters

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0' # Tensor name representing the outcome of the bottleneck layer in the inception-v3 model
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0' # The name corresponding to the image input tensor 

# Test Data
file_path = './test4/1/'
#y_test = [1,1,1,1,1,1,1]
#,1,1,1,1,1,1,1,1,0,0,0

# Read data
#image_data = tf.gfile.FastGFile(file_path, 'rb').read()

# Evaluate
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)

with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        # Read the trained inception-v3 model
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read()) # Load the inception-v3 model and return the data input tensor and bottleneck layer output tensor

        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def( graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # Use inception-v3 to process images to obtain feature vectors
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        i = 0
        for root, dirs,files in os.walk(file_path):

            for file in files:
                i=i+1
                image_data = tf.gfile.FastGFile(os.path.join(root, file) , 'rb').read()
                bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
                # Compress a four-dimensional array into a one-dimensional array.
                #Since the fully connected layer has a batch dimension when inputting, use the list as input.
                bottleneck_values = [np.squeeze(bottleneck_values)]
                # load graph and variables
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the input placeholder from the diagram by name
                input_x = graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]

                # The tensors we want to evaluate
                predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

                # Collecting predicted values
                all_predictions = []

                all_predictions = sess.run(predictions, {input_x: bottleneck_values})
                #print(all_predictions)
                (filepath, tempfilename) = os.path.split(file)
                print( i,":",all_predictions,tempfilename)

                if (file[len(file) - 5] == '0'):# If the label is 0
                    if (all_predictions == 0):# If the label is 1
                        TP = TP + 1
                    else:
                        FN = FN + 1
                else :
                    if (all_predictions == 0):
                        FP = FP + 1
                    else:
                        TN = TN + 1
# Print correct rate if label is provided

if file_path is not None:

    print(TP,FN,FP,TN)

    y_test = TP+FP+FN+TN
    #correct_predictions = float(sum(all_predictions == y_test))
    #wrong_predictions = float(sum(all_predictions == y_test))
    precision = float( TP/(TP+FP))
    recall = float(TP/(TP + FN))
    Acc = float((TP+TN)/(TP+FP+FN+TN))
    TPR = float(TP/(TP+FN))#sensitivity召回率 （TPR，真阳性率，灵敏度，召回率）
    TNR = float(TN/(FP+TN))#specificity（TNR，真阴性率，特异度）

    FNR = float(FN/(TP+FN))#Missed diagnosis rate，（1-sensitivity）
    FPR = float(FP / (TN + FP) ) # False positive rate(1-specificity),假阳性率，误诊率
    #print(sum(all_predictions))
    print('\nTotal number of test examples: {}'.format(y_test))

    print('Accuracy: %.2f%%' % (Acc*100))

    print('precision: %.2f%%' % (precision*100))
    print('sensitivity/recall(TPR): %.2f%%' % (recall*100))
    #print('sensitivity(TPR):%.2f%%' % (TPR*100))
    print('specificity(TNR): %.2f%%' % (TNR*100))

    #print('specificity(TNR): {:g}'.format(TNR))

