import os.path
from torchvision import transforms
import cv2
import numpy as np
from Image_Processing import binary_img
from Image_Processing import get_threshold_keys
from PIL import Image


def label_region(bin_img, width, height):
    visited = np.zeros(shape=bin_img.shape, dtype=np.uint8)
    label_img = np.zeros(shape=bin_img.shape, dtype=np.uint8)
    label = 0
    for i in range(height):
        for j in range(width):
            # find the seed
            if bin_img[i][j] == 255 and visited[i][j] == 0:
                # visit
                visited[i][j] = 1
                label += 1
                label_img[i][j] = label
                # label
                label_from_seed(bin_img, visited, i, j, label, label_img)
    return label_img, label


# use the regional growth method to mark
def label_from_seed(bin_img, visited, i, j, label, out_img):
    directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    seeds = [(i, j)]
    height = bin_img.shape[0]
    width = bin_img.shape[1]
    while len(seeds):
        seed = seeds.pop(0)
        i = seed[0]
        j = seed[1]
        if visited[i][j] == 0:
            visited[i][j] = 1
            out_img[i][j] = label

        # mark with (i,j) as the starting point
        for direct in directs:
            cur_i = i + direct[0]
            cur_j = j + direct[1]
            # illegality
            if cur_i < 0 or cur_j < 0 or cur_i >= height or cur_j >= width:
                continue
            # have not visited
            if visited[cur_i][cur_j] == 0 and bin_img[cur_i][cur_j] == 255:
                visited[cur_i][cur_j] = 1
                out_img[cur_i][cur_j] = label
                seeds.append((cur_i, cur_j))


def get_region_area(label_img, label):
    count = {key: 0 for key in range(label + 1)}
    start_pt = {key: (0, 0) for key in range(label + 1)}
    height = label_img.shape[0]
    width = label_img.shape[1]
    for i in range(height):
        for j in range(width):
            key = label_img[i][j]
            count[key] += 1
            if count[key] == 1:
                start_pt[key] = (j, i)
    return count, start_pt


def draw_area_reslult(img, count, start_pt):
    draw = img.copy()
    for key in count.keys():
        if key > 0:
            pt = start_pt[key]
            x = pt[0]
            y = pt[1]
            area = count[key]
            if y < 20:
                y = 20
            cv2.putText(
                draw, str(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (128, 0, 128), 1
            )
    return draw


def max_key(dic):
    max_label = max(dic, key=dic.get)
    return max_label


def img_resize520(in_path, out_path, size):
    for file_name in os.listdir(in_path):
        img_path = os.path.join(in_path, file_name)
        img = Image.open(img_path)
        resize = transforms.Resize(size)
        img = resize(img)
        img.save(out_path + "/" + file_name)


def get_molecular(init_path, mask_path, value):
    for file_name in os.listdir(mask_path):
        file_path = os.path.join(mask_path, file_name)
        img = cv2.imread(file_path)
        h = img.shape[0]
        w = img.shape[1]
        # graying
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # binaryzation
        bin_img = binary_img.get_binary_img(gray_img)
        # Label each shaded section
        label_img, label = label_region(bin_img, w, h)
        count, start_pt = get_region_area(label_img, label)
        draw = draw_area_reslult(img, count, start_pt)
        count.pop(0)
        # the labels whose ratio of each shadow part to the largest area in an image is greater than the set threshold are obtained
        if len(count) == 0:
            continue
        list_keys = get_threshold_keys.get_keys(count, value)
        init_img_name = file_name[:-4]
        init_img_path = init_path + "/" + init_img_name
        init_img = cv2.imread(init_img_path)
        # obtain images that are larger than the threshold
        temp = 1
        for key in list_keys:
            white_img = np.full(img.shape, 255, dtype=np.uint8)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if label_img[i][j] == key:
                        white_img[i][j] = init_img[i][j]
            if not os.path.exists(".//result_img"):
                os.makedirs(".//result_img")
            cv2.imwrite("./result_img/" + str(temp) + init_img_name, white_img)
            temp = temp + 1
    print(
        "Processing done:The extracted chemical molecular structure diagrams are stored in the result_img folder"
    )
