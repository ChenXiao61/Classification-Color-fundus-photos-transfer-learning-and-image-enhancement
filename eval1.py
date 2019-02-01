'''
This code is to predict the number of specified pictures one by one, 
pay attention to the path after runs, provide the file_path, y_test tag value, 
output normal and abnormal fundus lesions
'''

import tensorflow as tf
import numpy as np

# Model catalog
CHECKPOINT_DIR = './runs/1546517609/checkpoints'#save model after training
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb'

# the parameters of inception-v3 model
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # Tensor name representing the outcome of the bottleneck layer in the inception-v3 model
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # The name corresponding to the image input tensor

# path of new test set
file_path = './11.JPG'

# read images
def restore_model(testPic):
    image_data = tf.gfile.FastGFile(testPic, 'rb').read()

    # evaluate
    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:
            # read the trained inception-v3 model
            with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # laod inception-v3 model，and return the data input tensor and the bottleneck layer output tensor
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

            # Use inception-v3 to process images to obtain feature vectors
            bottleneck_values = sess.run(bottleneck_tensor,
                                         {jpeg_data_tensor: image_data})
            # Compress a four-dimensional array into a one-dimensional array. Since the fully connected 
            #layer has a batch dimension when inputting, use the list as input.
            bottleneck_values = [np.squeeze(bottleneck_values)]

            # loading graph and variables
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the input placeholder from the diagram by name
            input_x = graph.get_operation_by_name(
                'BottleneckInputPlaceholder').outputs[0]

            # The tensors we want to evaluate
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

            # Collecting predicted values
            all_predictions = []
            all_predictions = sess.run(predictions, {input_x: bottleneck_values})
            return all_predictions

# print the result of prediction on one photograph
def application():
    testNum = input("input the number of test pictures:")#Enter the number of images
    for i in range(int(testNum)):
        testPic = input("the path of test picture:")#Enter the path of oneimages
        preValue = restore_model(testPic)
        if preValue == [0]:#0 represent normal by the label
            print("Normal")

        else:
            print("Existence pathology")



def main():
    application()

if __name__ =='__main__':
    main()

