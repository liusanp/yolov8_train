import os
from tqdm import tqdm
import json

# 读取文件夹下的所有txt文档，计算文档每行第一个空格前的数字数量
def count_first_number(file_path):
    data_folder = r'D:\Document\captcha_detection\pdf_layout\project-69-at-2024-09-26-09-23-d91519f4'
    train_folder = os.path.join(data_folder, "labels")
    # val_folder = os.path.join(data_folder, "val/labels")
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.txt')]
    # val_files = [f for f in os.listdir(val_folder) if f.endswith('.txt')]
    res = {}
    for file in tqdm(train_files):
        with open(os.path.join(train_folder, file), 'r') as f:
            for line in f:
                num = line.split(' ')[0]
                res[num] = res.get(num, 0) + 1
    # for file in tqdm(val_files):
    #     with open(os.path.join(val_folder, file), 'r') as f:
    #         for line in f:
    #             num = line.split(' ')[0]
    #             res[num] = res.get(num, 0) + 1
    print(res)
    
    labels = []
    label_file = os.path.join(data_folder, "classes.txt")
    with open(label_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    print(labels)
    real_res = {}
    for k, v in res.items():
        real_res[labels[int(k)]] = v
        
    print(real_res)
    
    
def count_tag_pic():
    data_folder = r'D:\Document\captcha_detection\pdf_layout\project-69-at-2024-09-24-08-58-fcbdb065'
    train_folder = os.path.join(data_folder, "labels")
    # val_folder = os.path.join(data_folder, "val/labels")
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.txt')]
    # val_files = [f for f in os.listdir(val_folder) if f.endswith('.txt')]
    labels = []
    label_file = os.path.join(data_folder, "classes.txt")
    with open(label_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    print(labels)
    
    res = {}
    for file in tqdm(train_files):
        with open(os.path.join(train_folder, file), 'r') as f:
            for line in f:
                num = line.split(' ')[0]
                label = labels[int(num)]
                if label in res:
                    res[label].append(file.replace('.txt', ''))
                else:
                    res[label] = [file.replace('.txt', '')]
    print(res)
    with open('./tag_pics.txt', 'w') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=4))
    
    
if __name__ == '__main__':
    count_first_number(None)
    # count_tag_pic()