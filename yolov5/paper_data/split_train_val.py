# coding:utf-8

import os
import random
import argparse

parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default='./Annotations', type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='./ImageSets/Main', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = os.open(txtsavepath + '/trainval.txt', os.O_RDWR | os.O_CREAT)
file_test = os.open(txtsavepath + '/test.txt', os.O_RDWR | os.O_CREAT)
file_train = os.open(txtsavepath + '/train.txt', os.O_RDWR | os.O_CREAT)
file_val = os.open(txtsavepath + '/val.txt', os.O_RDWR | os.O_CREAT)

# for i in list_index:
#     name = total_xml[i][:-4] + '\n'
#     if i in trainval:
#         file_trainval.write(name)
#         if i in train:
#             file_train.write(name)
#         else:
#             file_val.write(name)
#     else:
#         file_test.write(name)

for i in list_index:
    # name = total_xml[i][:-4] + '\n'
    name = ".".join((os.path.basename(total_xml[i]).split('.')[0:-1])) + '\n'
    name.replace(' ','')
    if i in trainval:
        os.write(file_trainval,str.encode(name))
        if i in train:
            os.write(file_train,str.encode(name))
        else:
            os.write(file_val,str.encode(name))
    else:
        os.write(file_test,str.encode(name))

# file_trainval.close()
# file_train.close()
# file_val.close()
# file_test.close()

os.close(file_trainval)
os.close(file_train)
os.close(file_val)
os.close(file_test)