from ultralytics import YOLO


def predict_all():
    # 加载模型
    model = YOLO('runs/train-cls/book_card_cls/weights/best.pt')  # 加载自定义模型

    # 验证模型
    results = model.predict(source='data/images/G32', save=True, project='runs/predict-cls', name='book_card_cls')
    
    
def predict():
    # 加载模型
    model = YOLO('runs/train-cls/book_card_cls/weights/best.pt')  # 加载自定义模型

    # 验证模型
    results = model.predict(source='data/images/G32/G32 (99).jpg')
    print(results[0].probs.top1)
    
    
if __name__ == '__main__':
    # predict_all()
    predict()