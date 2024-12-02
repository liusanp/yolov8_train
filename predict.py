from ultralytics import YOLO
import matplotlib.pyplot as plt


def predict_all():
    # 加载模型
    model = YOLO('runs/train/book_card_tags2/weights/best.pt')  # 加载自定义模型

    # 验证模型
    results = model.predict(source='data/images/G32', save=True, project='runs/predict', name='book_card_tags2')
    
    
def predict():
    # 加载模型
    model = YOLO(r'D:\Document\captcha_detection\special_symbol\runs\train\20241121\train-special_symbol-20241121\weights\best.pt')  # 加载自定义模型
    img_path = r'D:\Document\captcha_detection\special_symbol\train_data\special_symbol-20241121\test\images\e4338bf5-241114-ZHY_15_NN_NN_04_03_03.jpg'
    # 验证模型
    results = model.predict(source=img_path)
    # print(results[0].names)
    # print(results[0].boxes.cls)
    # print(results[0].boxes.conf)
    # print(results[0].boxes.xyxyn)
    # print(results[0].orig_shape)
    image = plt.imread(img_path)
    height, width = image.shape[:2]
    # print(height, width)
    # 画图
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    boxes_pixel = results[0].boxes.xyxyn.clone()
    boxes_pixel[:, [0, 2]] *= width
    boxes_pixel[:, [1, 3]] *= height

    res = []
    # 画框
    idx = 0
    for box, class_index in zip(boxes_pixel, results[0].boxes.cls):
        xmin, ymin, xmax, ymax = box.tolist()
        name = (results[0].names)[class_index.item()]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(xmin, ymin - 0.05, name, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        
        confidence = (results[0].boxes.conf)[idx]
        idx += 1
        res.append({
            'name': name, 
            'confidence': round(float(confidence), 3), 
            'xmin': round(xmin, 4), 'ymin': round(ymin, 4), 
            'xmax': round(xmax, 4), 'ymax': round(ymax, 4),
            'class': int(class_index.item())
            })

    # 显示图片
    plt.show()
    # 保存图片
    # plt.savefig('result.jpg', dpi=100)
    print(res)
    
    
if __name__ == '__main__':
    # predict_all()
    predict()