'''代码使用说明：

确认y_true和y_scores文件夹是否需要清空
1注意runs的目录
2代码的输入是图片的路径file_path，图片格式为**0.JPG或者**1.JPG，即带有标签的图片名
3用之前一定要给图片改名字，最后一位需要带上0/1的标签
4得到两个文件夹，分别是真是标签和预测概率，用于画ROC曲线,还要注意两个文件有没有顶行写
'''
import tensorflow as tf
import numpy as np
import os
# 模型目录
CHECKPOINT_DIR = './runs/1546940151/checkpoints'
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb' # inception-v3模型参数

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0' # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0' # 图像输入张量对应的名称 # 测试数据


file_path = './test4/1/'

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
        i=0
        for root, dirs,files in os.walk(file_path):

            for file in files:
                i = i+1
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
                predictions2 = graph.get_operation_by_name('final_training_ops/Softmax').outputs[0]
                #print(predictions2)

                # 收集预测值
                all_predictions = []

                all_predictions = sess.run(predictions, {input_x: bottleneck_values})
                all_predictions_Prob = sess.run(predictions2, {input_x: bottleneck_values})

                (filepath, tempfilename) = os.path.split(file)
                if (file[len(file) - 5] == '0'):  # 如果标签为0
                    f_true = open('./y_true.txt', 'a')
                    f_true.write('0' + '\n')
                    f_true.close()

                    f_score = open('./y_scores.txt','a')
                    f_score.write(str(all_predictions_Prob[0][1]) + '\n')
                    f_score.close()
                if (file[len(file) - 5] == '1'):  # 如果标签为1
                    f_true = open('./y_true.txt', 'a')
                    f_true.write('1' + '\n')
                    f_true.close()

                    f_score = open('./y_scores.txt', 'a')
                    f_score.write(str(all_predictions_Prob[0][1]) + '\n')
                    f_score.close()


                #print(all_predictions2)
                print(i,":" ,tempfilename,'Real:',all_predictions,'Prediction:',all_predictions_Prob[0][0],'Prediction:',all_predictions_Prob[0][1])
        print('Number of samples:',i)
