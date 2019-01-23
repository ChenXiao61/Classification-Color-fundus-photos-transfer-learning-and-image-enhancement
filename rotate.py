'''import cv2
import numpy as np

image_source = cv2.imread("./0/1.JPEG")  # 读取图片
image= cv2.flip(image_source, 0, dst=None)  # 垂直镜像
img270=np.rot270(image)
cv2.imwrite("./test2/rot270.JPG", img270)  # 重命名并且保存

'''


from PIL import Image
import matplotlib.pyplot as plt
image1=Image.open("./data/fundus_photo/0/0_0.JPG")
im2 = image1.rotate(90)#可以更改90实现任意角度的旋转
im2.save("./19.JPG")
#plt.show()
print("处理完成")