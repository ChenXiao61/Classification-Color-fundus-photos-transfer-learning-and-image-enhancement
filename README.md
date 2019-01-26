# Classification-Color-fundus-photos-transfer-learning-and-image-enhancement
We combined the image enhancement algorithm with transfer learning to achieve the classification of color fundus photo.


首先对数据进行删除、裁剪处理

然后使用batchSplitGchannel.py对预处理后的图片提取绿通道，再运行CLAHE.py进行限制对比度自适应直方图均衡化

Inception-V3模型参数下载网址：https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

下载以后有两个文件1.imagenet_comp_graph_label_strings.txt 2.tensorflow_inception_graph.pb

将模型参数下载好之后model/ 文件夹下，还需要新建一个 tmp/bottleneck/ 文件夹用于存放每张图片通过 Inception-v3 模型计算得到的特征向量。

目录结构如下：

transfer-learning/

data/  
    fundus_photos/       
        0/   #正常眼底图片           
        1/   #异常眼底图片            
    tmp/      
        bottleneck/          
            ......              
model/   
    imagenet_comp_graph_label_strings.txt     
    tensorflow_inception_graph.pb     
train.py
train.py用于模型训练

eval1.py用于模型单张预测

eval2.py用于批量预测计算准确度，特异度，灵敏度

eval3.py用于保存模型预测每张的概率，画出ROC曲线和计算AUC



