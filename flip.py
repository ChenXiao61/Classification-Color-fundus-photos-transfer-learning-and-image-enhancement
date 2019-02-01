import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

source_path = "./0/"  
target_path = "./V_flip/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  

i = 0
for file in image_list:
    i = i + 1
    image_source = cv2.imread(source_path + file)  
    image= cv2.flip(image_source, 0, dst=None)  # 1 is horizontal flip, 0 is vertical flip, and -1 is diagonal mirror
    cv2.imwrite(target_path+file,image)
print("Finished")
