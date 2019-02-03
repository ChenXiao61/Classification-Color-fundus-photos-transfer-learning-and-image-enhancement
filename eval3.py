'''
Code usage instructions:
1  Confirm if the y_true and y_scores folders need to be emptied
2 Note the directory of runs
3 The input of the code is the path of the image file_path,
  the image format is **0.JPG or **1.JPG, that is, the name of the image with the label
4 Be sure to change the name of the picture before use. The last one needs to carry the label 0 or 1.
5 Get two folders, which are really labels and prediction probabilities, used to draw ROC curves,
  but also pay attention to whether the two files have top lines to write
'''
import tensorflow as tf
import numpy as np
import os
# Model catalog
CHECKPOINT_DIR = './runs/1546940151/checkpoints'
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb' # Inception-v3 model parameters

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0' 
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0' 
file_path = './test4/1/'
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)

with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def( graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        i = 0
        for root, dirs,files in os.walk(file_path):
            for file in files:
                i = i+1
                image_data = tf.gfile.FastGFile(os.path.join(root, file) , 'rb').read()
                bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
                bottleneck_values = [np.squeeze(bottleneck_values)]
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                input_x = graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]
                predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]
                predictions2 = graph.get_operation_by_name('final_training_ops/Softmax').outputs[0]
                #print(predictions2)

                all_predictions = []

                all_predictions = sess.run(predictions, {input_x: bottleneck_values})
                all_predictions_Prob = sess.run(predictions2, {input_x: bottleneck_values})

                (filepath, tempfilename) = os.path.split(file)
                if (file[len(file) - 5] == '0'):  
                    f_true = open('./y_true.txt', 'a')
                    f_true.write('0' + '\n')
                    f_true.close()

                    f_score = open('./y_scores.txt','a')
                    f_score.write(str(all_predictions_Prob[0][1]) + '\n')
                    f_score.close()
                if (file[len(file) - 5] == '1'): 
                    f_true = open('./y_true.txt', 'a')
                    f_true.write('1' + '\n')
                    f_true.close()

                    f_score = open('./y_scores.txt', 'a')
                    f_score.write(str(all_predictions_Prob[0][1]) + '\n')
                    f_score.close()


                #print(all_predictions2)
                print(i,":" ,tempfilename,'Real:',all_predictions,'Prediction:',all_predictions_Prob[0][0],'Prediction:',all_predictions_Prob[0][1])
        print('Number of samples:',i)
