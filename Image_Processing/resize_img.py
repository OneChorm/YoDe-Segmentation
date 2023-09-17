import os

from PIL import Image
from torchvision import transforms
def resize_img520(in_path,out_path):
    resize = transforms.Resize(520)
    paths = os.listdir(in_path)
    for img_name in paths:
        img_path = os.path.join(in_path, img_name)
        img = Image.open(img_path)
        img = resize(img)
        img.save(out_path+'/'+img_name)
