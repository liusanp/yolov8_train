import json
from pprint import pprint
from tqdm import tqdm
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime


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
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
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
    
    
def spilit_detect_data_by_tagsCate(data_folder, tags_cate):
    # 获取所有标签（所有标签在同一个lablestudio项目中标注）
    labels = []
    label_file = os.path.join(data_folder, "classes.txt")
    all_images_folder = os.path.join(data_folder, "images")
    all_labels_folder = os.path.join(data_folder, "labels")
    with open(label_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                labels.append(line)
    time_str = datetime.now().strftime("%Y%m%d")
    for k, v in tqdm(tags_cate.items()):
        # pdf_layout_table-20240926
        k_folder = os.path.join(data_folder, f"pdf_layout_{k}-{time_str}")
        os.makedirs(k_folder, exist_ok=True)
        # 写入classes.txt
        label_file = os.path.join(k_folder, "classes.txt")
        label_idx_link = {}
        with open(label_file, 'w', encoding='utf8') as f:
            new_idx = 0
            for i in v:
                f.write(labels[i] + "\n")
                label_idx_link[str(i)] = str(new_idx)
                new_idx += 1
        # 新建images和labels文件夹
        images_folder = os.path.join(k_folder, "images")
        labels_folder = os.path.join(k_folder, "labels")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)
        # 将有v中标签的图片和标注结果复制过来
        image_files = [f for f in os.listdir(all_images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
        for img in image_files:
            txt_path = os.path.splitext(img)[0] + ".txt"
            is_need_move = False
            new_label_res = []
            with open(os.path.join(all_labels_folder, txt_path), 'r') as f:
                for line in f.readlines():
                    label = line.strip().split(' ')[0]
                    if int(label) in v:
                        is_need_move = True
                        new_label_res.append(label_idx_link[label] + line.strip()[len(label):])
            if is_need_move:
                move_files(all_images_folder, images_folder, [img])
                with open(os.path.join(labels_folder, txt_path), 'w', encoding='utf8') as f:
                    for i in new_label_res:
                        f.write(i + "\n")
        spilit_detect_data_by_cate(k_folder)
        
        
def del_data_label(data_folder, del_cates):
    # 删除标注数据中的标签
    labels = []
    del_cates_idx = []
    label_file = os.path.join(data_folder, "classes.txt")
    all_images_folder = os.path.join(data_folder, "images")
    all_labels_folder = os.path.join(data_folder, "labels")
    with open(label_file, 'r', encoding='utf8') as f:
        cate_idx = 0
        for line in f.readlines():
            line = line.strip()
            if line and line not in del_cates:
                labels.append(line)
            else:
                del_cates_idx.append(cate_idx)
            cate_idx += 1
    print(labels)
    print(del_cates_idx)
    with open(label_file, 'w', encoding='utf8') as f:
        for line in labels:
            f.write(line + "\n")
    #  删除标签
    labels_files = [f for f in os.listdir(all_labels_folder) if f.endswith(('.txt'))]
    del_imgs = []
    for lab in tqdm(labels_files):
        labels_res = []
        with open(os.path.join(all_labels_folder, lab), 'r', encoding='utf8') as f:
            for line in f.readlines():
                label_split = line.strip().split(' ')
                label = int(label_split[0])
                new_label = label
                if label not in del_cates_idx:
                    for d in del_cates_idx:
                        if label > d:
                            new_label -= 1
                    labels_res.append(f'{new_label} {" ".join(label_split[1:])}\n')
        if labels_res:
            with open(os.path.join(all_labels_folder, lab), 'w', encoding='utf8') as f:
                for line in labels_res:
                    f.write(line)
        else:
            os.remove(os.path.join(all_labels_folder, lab))
            del_imgs.append(os.path.splitext(lab)[0])
    for f in os.listdir(all_images_folder):
        if os.path.splitext(f)[0] in del_imgs:
            os.remove(os.path.join(all_images_folder, f))


if __name__ == '__main__':
    # 分类训练集处理
    # do_handle()
    # 目标检测训练集处理
    # spilit_detect_data()
    # 切分训练集
    spilit_detect_data_by_cate(r'D:\Document\captcha_detection\special_symbol\train_data\special_symbol-20241202')
    # 切分训练集，按标签类别区分
    # spilit_detect_data_by_tagsCate(
    #     r'D:\Document\captcha_detection\pdf_layout\train_data\pdf_layout-20241106', 
    #     {
    #         "table": [9],
    #         "formula": [5],
    #         "anno": [0, 1],
    #         "bold": [2],
    #         "italic": [6],
    #         "subScript": [7],
    #         "superScript": [8],
    #         "textattr": [3, 4, 10],
    #     }
    # )
    # 去除训练集标签  ccc6  hui20
    # del_data_label(r'D:\Document\captcha_detection\special_symbol\train_data\special_symbol-20241202', ['ccc', 'hui', 'dcf', 'dcz', 'dc', 'wdcf', 'wdcz', 'wdcwzf'])
