
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

source_path = "./2Jan/0/"  #  path of source file(according to your own images path)

target_path = "./2Jan/0_G_Channel_CLAHE/"#  path of target file（Defined by yourself）

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # obtain the images in the source file

for file in image_list: 
    image_source = cv2.imread(source_path + file)  # read the image in the source file
    b, g, r = cv2.split(image_source)# split the R,G,B channel of image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)#do clahe for the green channel of the image
    cv2.imwrite(target_path + file ,g)#write into target file path after processing
print("All images Finished")
