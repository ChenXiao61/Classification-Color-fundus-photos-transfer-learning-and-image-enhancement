import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

source_path = "./2Jan/1/"  # path of source file
target_path = "./2Jan/1_G_Channel/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # obtain the image in the source file

i = 0
for file in image_list:
    i = i + 1
    RGB = cv2.imread(source_path + file, -1)  # read one image from source file
    b, g, r = cv2.split(RGB)  # apilt the R,G,B channel of the image
    cv2.imwrite(target_path+file, g)  # write into target file
print("Finished")
