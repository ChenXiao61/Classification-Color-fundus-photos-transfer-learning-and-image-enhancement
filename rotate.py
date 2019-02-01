import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

source_path = "./0/" 
target_path = "./r315/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path) 

i = 0
for file in image_list:
    i = i + 1
    image = Image.open(source_path + file)
    im2 = image.rotate(315)
    im2.save(target_path+file)  # 
print("Finished")
