'''代码使用说明:

    xiugai  range  de daxiao 给出模型的标签和对应分类的预测概率的文件地址，即可画出ROC 曲线

'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc

###################################################################
f = open( './y_true.txt')
y_true = []
for i in range(100):  # 383 = the number of images in test set
    line = f.readline()
    y_true = y_true + [float(line)]

f.close()
print('yy_true:', y_true)
print('length of y_true:', y_true.__len__())

###################################################################
f = open('./y_scores.txt')
y_scores = []
for i in range(100):
    line = f.readline()
    y_scores = y_scores + [float(line)]

f.close()
print('yy_scores:', y_scores)
print('length of yy_true:', y_scores.__len__())

print('finish loading yy_true  yy_scores')

###################################################################
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
AUC = auc(fpr, tpr)

print('fpr:')
print(fpr)
print(fpr.__len__())

print('tpr:')
print(tpr)
print(tpr.__len__())

###################################################################
plt.plot(fpr, tpr)
plt.title('ROC_curve' + '(AUC: ' + str(AUC) + ')')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
