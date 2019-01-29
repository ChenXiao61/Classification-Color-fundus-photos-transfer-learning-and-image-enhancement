
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

image_size = 224  # 设定尺寸
source_path = "./2Jan/0/"  # 源文件路径

target_path = "./2Jan/0_G_Channel_CLAHE/"#目标文件路径

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # 获得文件名

for file in image_list: 
    #image_source = Image.open(source_path + file)
    image_source = cv2.imread(source_path + file)  # 读取图片'''
    #image = cv2.imread("./1.JPEG", cv2.IMREAD_COLOR)
    #cv2.imwrite('image.JPG', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b, g, r = cv2.split(image_source)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
#r = clahe.apply(r)
#image = cv2.merge([b, g, r])
#cv2.imwrite('clahe.JPG', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
    cv2.imwrite(target_path + file ,g)

print("批量处理完成")
