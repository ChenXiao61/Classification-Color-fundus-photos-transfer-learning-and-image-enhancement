from PIL import Image
import os
from PIL import ImageEnhance
source_path = './bright/'#the path of source file（according to your own images path）
target_path = './bright1/'#the path of target file

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # obtain the image from source_path

for file in image_list:
    img = Image.open(source_path + file)#open the image in the source_path
    imgGray =img.convert('L')#convert original image into grayscale
    #img.show()
    allpixel = 0
    for x in range(imgGray.size[0]):
        for y in range(imgGray.size[1]):
            pixel = imgGray.getpixel((x, y))#obtain the value of current location
            allpixel = allpixel + pixel#calculate all pixel in one image
    print(allpixel)
    if allpixel < 400000000:
        img.convert("RGB")
        enh_bri = ImageEnhance.Brightness(img) # enhance the original image brightness
        brightness = 2
        image_brightened = enh_bri.enhance(brightness)
        image_brightened.save(target_path+file)  # save the imge into targe file
print("Finished!")

