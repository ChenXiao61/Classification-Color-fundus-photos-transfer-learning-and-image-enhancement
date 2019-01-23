import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

image_size = 224  # 设定尺寸
source_path = "./0/"  # 源文件路径
#target_path = "./1/rotate45/"  # 输出目标文件路径
#target_path = "./1/rotate90/"  # 输出目标文件路径
#target_path = "./1/rotate135/"  # 输出目标文件路径
#target_path = "./1/rotate225/"  # 输出目标文件路径
#target_path = "./1/rotate270/"  # 输出目标文件路径
target_path = "./V_flip/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # 获

# 得文件名

i = 0
for file in image_list:
    i = i + 1
    #image_source = Image.open(source_path + file)
    image_source = cv2.imread(source_path + file)  # 读取图片
    image= cv2.flip(image_source, 0, dst=None)  # 1为水平翻转，0为垂直镜像，-1为对角镜像(buyong)
    cv2.imwrite(target_path+file,image)
    #im2 = image.rotate(315)
    #image.save(target_path+file)  # 重命名并且保存
print("批量处理完成")