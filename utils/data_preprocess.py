import re
import os
from PIL import Image

from datasets import Dataset

def img_dir_2_dict(directory):
    images = dict()
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            images[filename[:-4]] = image
    
    return images

def split_label_dict(file):
    # 正则表达式解释：
    # (.*) 表示匹配任意字符（除换行符）0次或多次，并捕获这部分内容。
    # \.png 匹配字符串 ".png"，注意\.用于匹配实际的点字符，因为点在正则表达式中是特殊字符。
    pattern = r'(.*)(\.png\t)(.*)'
    labels = dict()

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            match = re.match(pattern, line)
            if match:
                before_png, png, after_png = match.groups()
                labels[before_png] = after_png
    
    return labels

def dict_dataset(images, labels):
    file_name = list(images.keys())
    image = [images[name] for name in file_name]
    label = [labels[name] for name in file_name]
    dataset = {'name': file_name, 'image': image, 'label': label}

    return dataset