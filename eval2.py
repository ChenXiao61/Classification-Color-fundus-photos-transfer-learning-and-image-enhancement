#代码使用说明：
'''
1注意runs的目录
2代码的输入是图片的路径，图片格式为**0.JPG或者**1.JPG，即带有标签的图片名

'''

import tensorflow as tf
import numpy as np
import os
# 模型目录
CHECKPOINT_DIR = './runs/1547178289/checkpoints'
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb' # inception-v3模型参数

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0' # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0' # 图像输入张量对应的名称 # 测试数据

#file_path = './data/flower_photos/tulips/11746080_963537acdc.jpg'
#y_test = [4]

file_path = './test4/1/'
#y_test = [1,1,1,1,1,1,1]
#,1,1,1,1,1,1,1,1,0,0,0

# 读取数据
#image_data = tf.gfile.FastGFile(file_path, 'rb').read()

# 评估
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)

with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        # 读取训练好的inception-v3模型
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read()) # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量

        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def( graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 使用inception-v3处理图片获取特征向量
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
                # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
                bottleneck_values = [np.squeeze(bottleneck_values)]
                # 加载元图和变量
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # 通过名字从图中获取输入占位符
                input_x = graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]

                # 我们想要评估的tensors
                predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

                # 收集预测值
                all_predictions = []

                all_predictions = sess.run(predictions, {input_x: bottleneck_values})
                #print(all_predictions)
                (filepath, tempfilename) = os.path.split(file)
                print( i,":",all_predictions,tempfilename)

                if (file[len(file) - 5] == '0'):#如果标签为0
                    if (all_predictions == 0):#如果预测为0
                        TP = TP + 1
                    else:
                        FN = FN + 1
                else :
                    if (all_predictions == 0):
                        FP = FP + 1
                    else:
                        TN = TN + 1
# 如果提供了标签则打印正确率

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

    FNR = float(FN/(TP+FN))#漏诊率，（1-sensitivity）
    FPR = float(FP / (TN + FP) ) # 假正例率(1-specificity),假阳性率，误诊率
    #print(sum(all_predictions))
    print('\nTotal number of test examples: {}'.format(y_test))

    print('Accuracy: %.2f%%' % (Acc*100))

    print('precision: %.2f%%' % (precision*100))
    print('sensitivity/recall(TPR): %.2f%%' % (recall*100))
    #print('sensitivity(TPR):%.2f%%' % (TPR*100))
    print('specificity(TNR): %.2f%%' % (TNR*100))

    #print('specificity(TNR): {:g}'.format(TNR))

