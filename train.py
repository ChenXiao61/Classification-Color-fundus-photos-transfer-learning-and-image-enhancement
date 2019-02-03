#tensorboard --logdir=train:"./runs/1546517609/summaries/train",val:"./runs/1546517609/summaries/dev"
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import math

MODEL_DIR = './model/'
MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = './data/tmp/bottleneck'
INPUT_DATA = './data/fundus_photo'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


BOTTLENECK_TENSOR_SIZE = 2048  # Number of nodes in the inception-v3 model bottleneck layer
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # Tensor name representing the outcome of the bottleneck layer in the inception-v3 model
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  #The name corresponding to the image input tensor

# Training parameters of neural network
BASE_LEARNING_RATE = 0.1
LEARNING_RATE = 0.01
#LEARNING_RATE_BASE = 0.01
MIN_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.99
STEPS = 50000

BATCH = 256

CHECKPOINT_EVERY = 100
NUM_CHECKPOINTS = 5


# Read all the image lists from the data folder and separate them by training set, validation set, and test set.
def create_image_lists(validation_percentage, test_percentage):
    result = {}  # Save all images. Key is the category name. Value is also a dictionary, storing all the name of images 
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # Get all subdirectories
    is_root_dir = True  # The first directory is the current directory and needs to be ignored.

    # Operate each subdirectory separately
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # Get all valid images in the current directory
        extensions = {'jpg', 'jpeg', 'JPG', 'JPEG'}
        file_list = []  # Save all images
        dir_name = os.path.basename(sub_dir)  # Get the last directory name of the path
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # Randomly divide the pictures of the current category into training set, test set, and validaation set.
        label_name = dir_name.lower()  # Get the name of the category by directory name
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # Get the name of the image
            chance = np.random.randint(100)  # Randomly generate 100 numbers to represent percentage
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (validation_percentage + test_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # Put the data set of the current category into the result dictionary
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    # Return all the data that has been sorted out
    return result


# Get the address of an image by category name, dataset, and image number
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # Get all the images in a given category
    category_list = label_lists[category]  # Get all the images in the collection based on the name of the dataset it belongs to
    mod_index = index % len(category_list)  # Normalize the index of image
    base_name = category_list[mod_index]  # Get the file name of the image
    sub_dir = label_lists['dir']  # Get the directory name of the current category
    full_path = os.path.join(image_dir, sub_dir, base_name)  # Absolute path of the image
    return full_path


# Get the address of the feature vector value by category name, data set, and image number
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index,
                          category) + '.txt'
# Use inception-v3 to process images to obtain feature vectors
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)  # Compress a four-dimensional array into a one-dimensional array
    return bottleneck_values


# Get the feature vector of an image processed by the inception-v3 model
def get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                             jpeg_data_tensor, bottleneck_tensor):
    # Get the path of the feature vector file corresponding to a image
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)

    # If the feature vector file does not exist, it is calculated and saved by the inception-v3 model.
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index,
                                    category)  # Get the original path of the image
        image_data = gfile.FastGFile(image_path, 'rb').read()  # Get image content
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor,
            bottleneck_tensor)  # Calculating feature vector by inception-v3

        # Save the feature vector to a file
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # Otherwise, get the feature vector of the image directly from the file.
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    # Return the resulting feature vector
    return bottleneck_values


# Randomly get a batch image as training data
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # Randomly add a category and image number to the current training data
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# Get all the test data
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                         bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # Enumerate all categories and test images in each category
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(
                image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category,
                jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def main(_):
    global LEARNING_RATE
    # Read all the images
    image_lists = create_image_lists(VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    n_classes = len(image_lists.keys())
    global_step = tf.Variable(0,trainable=False)

    with tf.Graph().as_default() as graph:
        # Read the trained inception-v3 model
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # Load the inception-v3 model and return the data input tensor and bottleneck layer output tensor
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME
                ])

        # Define new neural network inputs
        bottleneck_input = tf.placeholder(
            tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        # Define a new standard answer input
        ground_truth_input = tf.placeholder(
            tf.float32, [None, n_classes], name='GroundTruthInput')

        # Define a layer of fully connected layers to solve new image classification problems
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.1))#stddev=0.001
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        # Define the cross entropy loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,global_step,(All_Example*0.8)/BATCH,LEARNING_RATE_DECAY)

        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        #train_step = tf.train.AdamOptimizer().minimize(
            cross_entropy_mean)

        # Calculation accuracy
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(
                tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))



    # Training process
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer().run()
        # Save the catalog of models and summary
        import time
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, 'runs', timestamp))
        print('\nWriting to {}\n'.format(out_dir))
        # Summary of loss value and correct rate
        loss_summary = tf.summary.scalar('loss', cross_entropy_mean)
        acc_summary = tf.summary.scalar('accuracy', evaluation_step)
        # train summary
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir+'/train',
                                                     sess.graph)
        # validation summary
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        #dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir+'/val')


        # save checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)

        for i in range(STEPS):
            # Get training data for one batch each time
            LEARNING_RATE = BASE_LEARNING_RATE / (math.exp(
                i / (STEPS / math.log(0.1 / MIN_LEARNING_RATE))))
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH,
                                                                                  'training',jpeg_data_tensor, bottleneck_tensor)
            _, train_summaries = sess.run([train_step, train_summary_op],feed_dict={
                    bottleneck_input: train_bottlenecks,
                    ground_truth_input: train_ground_truth
                })


            # Save a summary of each step
            train_summary_writer.add_summary(train_summaries, i)


            # Test correct rate on the validation set
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes,
                                                                                                image_lists, BATCH,
                                                                                                'validation',
                                                                                                jpeg_data_tensor,
                                                                                                bottleneck_tensor)
                validation_accuracy, dev_summaries = sess.run([evaluation_step, dev_summary_op],
                                                              feed_dict={
                                                                  bottleneck_input: validation_bottlenecks,
                                                                  ground_truth_input: validation_ground_truth
                                                              })
                print('Step %d : Validation accuracy on random sampled %d examples = %.1f%%,Learning Rate: %lf' % (
                    i, BATCH, validation_accuracy * 100, LEARNING_RATE))


            '''va = str(validation_accuracy)
                    f = open('./validation.txt','a')
                    f.writelines(va+'\n')
                    f.close()'''



             #Save the model and test summary every CHECKPOINT_EVERY
            if i % CHECKPOINT_EVERY == 0:
                dev_summary_writer.add_summary(dev_summaries, i)
                path = saver.save(sess, checkpoint_prefix, global_step=i)

                #print('Saved model checkpoint to {}\n'.format(path))

        # Finally test the correct rate on the test set
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={
                bottleneck_input: test_bottlenecks,
                ground_truth_input: test_ground_truth
            })
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

        # save label
        output_labels = os.path.join(out_dir,'label.txt')
        with tf.gfile.FastGFile(output_labels, 'w') as f:
            keys = list(image_lists.keys())
            for i in range(len(keys)):
                keys[i] = '%2d -> %s' % (i, keys[i])
            f.write('\n'.join(keys) + '\n')




if __name__ == '__main__':
    tf.app.run()
