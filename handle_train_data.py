import json
from pprint import pprint
from tqdm import tqdm
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


# 处理label studio导出的分类打标图片

def read_json():
    json_path = r"D:\Document\captcha_detection\book_card\project-38-at-2024-01-04-10-34-4bf1553a.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_img_to_dir(label_data):
    # 图片文件夹路径
    image_folder = r"D:\Document\captcha_detection\book_card\38"
    # 设置划分比例
    train_ratio = 0.8  # 80% 划分到 "train" 文件夹
    # 创建 "train" 和 "test" 文件夹
    train_folder = os.path.join(image_folder, "train")
    test_folder = os.path.join(image_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    # 遍历 JSON 数据
    for filename, label in tqdm(label_data.items()):
        # 构建源文件路径
        source_path = os.path.join(image_folder, filename)
        # 确定目标文件夹路径
        target_folder = train_folder if hash(filename) % 100 < train_ratio * 100 else test_folder
        target_label_folder = os.path.join(target_folder, label)
        # 创建目标标签文件夹（如果不存在）
        os.makedirs(target_label_folder, exist_ok=True)
        # 构建目标文件路径
        target_path = os.path.join(target_label_folder, filename)
        # 移动文件
        shutil.move(source_path, target_path)
    

def do_handle():
    json_data = read_json()
    # pprint(json_data)
    label_data = {}
    for j in tqdm(json_data):
        file_name = j['file_upload']
        labels = j['annotations'][0]['result'][0]['value']['choices']
        labels_str = ','.join(labels)
        label_data[file_name] = labels_str
    # pprint(label_data)
    save_img_to_dir(label_data)
    pass


def create_folders(base_folder, subfolders):
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        os.makedirs(folder_path, exist_ok=True)

def move_files(source_folder, target_folder, files):
    for file_name in files:
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        shutil.copy(source_path, target_path)


def spilit_detect_data():
    data_folder = r'D:\Document\captcha_detection\clothes_tags\train_data\clothes_tags-20240828'
    
    # 设置划分比例
    train_ratio = 0.9  # 80% 划分到 "train" 文件夹
    # 创建 "train" 和 "test" 文件夹
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    train_folder = os.path.join(data_folder, "train")
    val_folder = os.path.join(data_folder, "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    # Get the list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

    # Split the data into train and val sets
    train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)
    
    create_folders(train_folder, ["images", "labels"])
    create_folders(val_folder, ["images", "labels"])

    move_files(images_folder, os.path.join(train_folder, "images"), train_images)
    move_files(images_folder, os.path.join(val_folder, "images"), val_images)

    # Move corresponding labels to train and val folders
    move_files(labels_folder, os.path.join(train_folder, "labels"), [os.path.splitext(image)[0] + ".txt" for image in train_images])
    move_files(labels_folder, os.path.join(val_folder, "labels"), [os.path.splitext(image)[0] + ".txt" for image in val_images])
    
    
def count_first_number(file_path):
    label_folder = os.path.join(file_path, "labels")
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    res = {}
    for file in tqdm(label_files):
        with open(os.path.join(label_folder, file), 'r') as f:
            for line in f.readlines():
                num = line.strip().split(' ')[0]
                res[num] = res.get(num, 0) + 1
                
    labels = []
    label_file = os.path.join(file_path, "classes.txt")
    with open(label_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                labels.append(line)
    real_res = {}
    for k, v in res.items():
        real_res[labels[int(k)]] = v
    print(real_res)
    return len(res)
    
    
def spilit_detect_data_by_cate(data_folder):
    # data_folder = r'D:\Document\captcha_detection\pdf_layout\train_data\pdf_layout-20240913'
    cate_num = count_first_number(data_folder)
    print(f'cate_num: {cate_num}')
    # 设置划分比例
    train_ratio = 0.8  # 80% 划分到 "train" 文件夹
    val_ratio = 0.1
    test_ratio = 0.1
    # 创建 "train" 和 "test" 文件夹
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    train_folder = os.path.join(data_folder, "train")
    val_folder = os.path.join(data_folder, "val")
    test_folder = os.path.join(data_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    # Get the list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    val_count = int(len(image_files) * val_ratio)
    test_count = int(len(image_files) * test_ratio)
    train_count = len(image_files) - val_count - test_count
    print(f'total: {len(image_files)}, train_count: {train_count}, val_count: {val_count}, test_count: {test_count}')
    create_folders(train_folder, ["images", "labels"])
    create_folders(val_folder, ["images", "labels"])
    create_folders(test_folder, ["images", "labels"])
    # 验证集保证有所有类别
    val_cates = {}
    val_images = []
    for img in image_files:
        txt_path = os.path.splitext(img)[0] + ".txt"
        if len(val_cates) >= cate_num:
            if len(val_images) >= val_count:
                break
            else:
                val_images.append(img)
                move_files(images_folder, os.path.join(val_folder, "images"), [img])
                move_files(labels_folder, os.path.join(val_folder, "labels"), [txt_path])
                continue
        else:
            with open(os.path.join(labels_folder, txt_path), 'r') as f:
                for line in f.readlines():
                    label = line.strip().split(' ')[0]
                    if label not in val_cates:
                        val_images.append(img)
                        val_cates[label] = val_cates.get(label, 0) + 1
                        move_files(images_folder, os.path.join(val_folder, "images"), [img])
                        move_files(labels_folder, os.path.join(val_folder, "labels"), [txt_path])
                        break
    # 如果验证集不够数，补充
    if len(val_images) < val_count:
        for img in image_files:
            if len(val_images) >= val_count:
                break
            if img not in val_images:
                val_images.append(img)
                move_files(images_folder, os.path.join(val_folder, "images"), [img])
                move_files(labels_folder, os.path.join(val_folder, "labels"), [os.path.splitext(img)[0] + ".txt"])
    # 训练集保证有所有类别
    train_cates = {}
    train_images = []
    for img in image_files:
        if img in val_images:
            continue
        txt_path = os.path.splitext(img)[0] + ".txt"
        if len(train_cates) >= cate_num:
            if len(train_images) >= train_count:
                break
            else:
                train_images.append(img)
                move_files(images_folder, os.path.join(train_folder, "images"), [img])
                move_files(labels_folder, os.path.join(train_folder, "labels"), [txt_path])
                continue
        else:
            with open(os.path.join(labels_folder, txt_path), 'r') as f:
                for line in f.readlines():
                    label = line.strip().split(' ')[0]
                    if label not in train_cates:
                        train_images.append(img)
                        train_cates[label] = train_cates.get(label, 0) + 1
                        move_files(images_folder, os.path.join(train_folder, "images"), [img])
                        move_files(labels_folder, os.path.join(train_folder, "labels"), [txt_path])
                        break
    # 如果训练集不够数，补充
    if len(train_images) < train_count:
        for img in image_files:
            if len(train_images) >= train_count:
                break
            if img not in val_images and img not in train_images:
                train_images.append(img)
                move_files(images_folder, os.path.join(train_folder, "images"), [img])
                move_files(labels_folder, os.path.join(train_folder, "labels"), [os.path.splitext(img)[0] + ".txt"])
    # 测试集
    test_images = []
    for img in image_files:
        if img not in val_images and img not in train_images:
            test_images.append(img)
            move_files(images_folder, os.path.join(test_folder, "images"), [img])
            move_files(labels_folder, os.path.join(test_folder, "labels"), [os.path.splitext(img)[0] + ".txt"])
    print(f'total: {len(image_files)}, train_images: {len(train_images)}, val_images: {len(val_images)}, test_images: {len(test_images)}')


if __name__ == '__main__':
    # 分类训练集处理
    # do_handle()
    # 目标检测训练集处理
    # spilit_detect_data()
    spilit_detect_data_by_cate(r'D:\Document\captcha_detection\pdf_layout\train_data\pdf_layout-20240913')