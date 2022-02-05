import os,glob
import csv
import random

class_to_num = {}

class_name_list = os.listdir(r'C:\Users\GLT\Desktop\Nums')
# print(class_name_list)

for class_name in class_name_list:
    class_to_num[class_name] = len(class_to_num.keys())#所有类别的名字按顺序分配序号
# print(class_to_num)

image_dir = []
for class_name in class_name_list:#得到每个样本的路径
    image_dir +=glob.glob(os.path.join(r'C:\Users\GLT\Desktop\Nums',class_name,'*.png'))
# print(image_dir)
random.shuffle(image_dir)
# print(image_dir)
with open('myself_data.csv',mode='w',newline='')as f:
    writer = csv.writer(f)
    for image in image_dir:
        class_name = image.split(os.sep)[-2]#拿到路径里的类名，再把类名转换为之前定义好的序号，即为该图像的标签
        label = class_to_num[class_name]
        writer.writerow([image,label])
print('ok')