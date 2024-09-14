from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import torch
import csv


data_name = 'device_pack-20240912'
train_idx = ''
detect_idx = ''
# 加载 YOLOv8 模型
model = YOLO(f'./runs/train/train-{data_name}{train_idx}/weights/best.pt')
# model = YOLO(r'D:\Document\captcha_detection\clothes_tags\runs\train\train-clothes_tags-20240830\weights\best.pt')
# 测试集路径
test_path = f'./datasets/{data_name}/test/'
# test_path = r'D:\Document\captcha_detection\clothes_tags\train_data\clothes_tags-20240830\test'
save_path = f'./runs/detect/detect-{data_name}{detect_idx}/img_results.csv'
# save_path = './img_results.csv'


def test_label():
    # 加载测试数据集
    test_dataset = f'datasets/{data_name}/data.yaml'
    # 保存结果路径
    # save_path = './runs/predict/clothes_tags_20240731/'
    # 预测测试集
    # results = model(test_dataset, conf=0.7)
    results = model.val(data=test_dataset, split='test', save_json=True, plots=True, name=f'detect-{data_name}')
    # print(results)
    
    
def load_annotations(image_name, labels_dir, image_size):
    """
    从标注文件中加载真实框数据并转换为像素坐标。
    
    :param image_name: 图像文件名（不带扩展名）
    :param labels_dir: 存储标注文件的文件夹路径
    :param image_size: 图像的尺寸 (width, height)
    :return: 返回一个列表，包含每个真实框的 [class_id, x1, y1, x2, y2] 信息
    """
    labels_path = os.path.join(labels_dir, f"{image_name}.txt")
    boxes = []
    
    with open(labels_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # 转换为左上角和右下角的坐标
            x1 = (x_center - width / 2) * image_size[0]
            y1 = (y_center - height / 2) * image_size[1]
            x2 = (x_center + width / 2) * image_size[0]
            y2 = (y_center + height / 2) * image_size[1]
            
            boxes.append([int(class_id), x1, y1, x2, y2])
    return boxes


def box_iou(box1, box2):
    """
    计算两个矩形框的IoU
    :param box1: Tensor类型，形状为(N, 4)，表示N个预测框的坐标(x1, y1, x2, y2)
    :param box2: Tensor类型，形状为(M, 4)，表示M个真实框的坐标(x1, y1, x2, y2)
    :return: Tensor类型，形状为(N, M)，表示N个预测框与M个真实框的IoU值
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)


def evaluate_imagei_old(image_path, labels_dir, model, iou_threshold=0.6):
    """
    评估单张图片中所有标签是否都被正确预测。
    
    :param image_path: 图片文件路径
    :param labels_dir: 存储标注文件的文件夹路径
    :param model: YOLOv8模型
    :param iou_threshold: 用于判断预测是否正确的IoU阈值
    :return: 是否所有标签都被正确预测 (True/False)
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 载入图片的标注数据
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    image_size = (width, height)
    gt_boxes = load_annotations(image_name, labels_dir, image_size)
    
    # 使用YOLOv8模型进行推理
    results = model.predict(source=image_path)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_class_ids = results[0].boxes.cls.cpu().numpy()
    
    # 如果没有预测框，直接返回False
    if len(pred_boxes) == 0:
        return False
    
    gt_boxes = torch.tensor(gt_boxes)
    pred_boxes = torch.tensor(pred_boxes)[:, :4]  # 取出预测框的坐标
    pred_class_ids = torch.tensor(pred_class_ids)
    
    # 计算IoU
    ious = box_iou(pred_boxes, gt_boxes[:, 1:])  # 计算预测框与真实框之间的IoU

    # 评估每个预测框与真实框的IoU和class_id是否都匹配
    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    for i, pred_box in enumerate(pred_boxes):
        # 找到IoU大于阈值并且class_id匹配的真实框
        valid_matches = (ious[i] >= iou_threshold) & (pred_class_ids[i] == gt_boxes[:, 0])
        if torch.any(valid_matches):
            # matched_gt[valid_matches.argmax()] = True
            matched_gt[valid_matches.int().argmax()] = True

    return torch.all(matched_gt).item()


def evaluate_image(image_path, labels_dir, model, iou_threshold=0.6):
    """
    评估单张图片中所有标签是否都被正确预测，并返回相关信息。
    
    :param image_path: 图片文件路径
    :param labels_dir: 存储标注文件的文件夹路径
    :param model: YOLOv8模型
    :param iou_threshold: 用于判断预测是否正确的IoU阈值
    :return: 图片名, 标注框信息, 预测框信息, 标注框数, 预测正确的框数, 是否全对
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # 使用YOLOv8模型进行推理
    results = model.predict(source=image_path)
    height = results[0].orig_shape[0]
    width = results[0].orig_shape[1]
    # 载入图片的标注数据
    # image = plt.imread(image_path)
    # height, width = image.shape[:2]
    image_size = (width, height)
    gt_boxes = load_annotations(image_name, labels_dir, image_size)
    
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_class_ids = results[0].boxes.cls.cpu().numpy()
    # 所有标签
    names = results[0].names
    # 标注标签
    gt_labels = [names[int(idx[0])] for idx in gt_boxes]
    
    # 如果没有预测框，直接返回结果
    if len(pred_boxes) == 0:
        return image_name, str(gt_labels), "[]", len(gt_boxes), 0, False
    
    gt_boxes = torch.tensor(gt_boxes)
    pred_boxes = torch.tensor(pred_boxes)[:, :4]  # 取出预测框的坐标
    pred_class_ids = torch.tensor(pred_class_ids)
    # 预测标签
    pred_labels = [names[int(idx)] for idx in pred_class_ids]
    # 计算IoU
    ious = box_iou(pred_boxes, gt_boxes[:, 1:])  # 计算预测框与真实框之间的IoU

    # 评估每个预测框与真实框的IoU和class_id是否都匹配
    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    correct_count = 0
    for i, pred_box in enumerate(pred_boxes):
        # 找到IoU大于阈值并且class_id匹配的真实框
        valid_matches = (ious[i] >= iou_threshold) & (pred_class_ids[i] == gt_boxes[:, 0])
        if torch.any(valid_matches):
            matched_gt[valid_matches.int().argmax()] = True
            correct_count += 1
    
    all_correct = torch.all(matched_gt).item()
    return image_name, str(gt_labels), str(pred_labels), len(gt_boxes), correct_count, all_correct


def test_image():
    image_dir = os.path.join(test_path, "images")
    labels_dir = os.path.join(test_path, "labels")
    
    # CSV文件头
    csv_header = ["图片名", "标注", "预测", "标注数", "预测正确数", "是否全对"]

    all_correct = 0
    total_images = 0
    with open(save_path, mode='w', newline='', encoding='gb18030') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        for image_file in os.listdir(image_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                total_images += 1
                image_path = os.path.join(image_dir, image_file)
                result = evaluate_image(image_path, labels_dir, model)
                writer.writerow(result)
                if result[-1]:
                    all_correct += 1
                
    print(f"所有标注都预测正确的图片数: {all_correct}/{total_images}")


if __name__ == '__main__':
    test_label()
    test_image()
