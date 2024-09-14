from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('models/yolov8x-cls.pt')  # load a custom model
    # model = YOLO('runs/train-cls/product_cls_20240821/weights/best.pt')  # load a custom model
    results = model.train(
        # model='models/yolov8x-cls.pt', 
        # pretrained='/data/git/yolov8/models/yolov8l-cls.pt', 
        task='classify', 
        data='datasets/prod_cate', 
        epochs=300, 
        batch=24,
        device=4, 
        optimizer='SGD',
        momentum=0.9,
        lr0=0.001,
        # lrf=0.01,
        lrf=0.01,
        patience=0,
        amp=False, 
        project='runs/train-cls', 
        name='product_cls_20240821')

# optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
# optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 50 weight(decay=0.0), 51 weight(decay=0.0005625000000000001), 51 bias(decay=0.0)

# - prod_cate
# --- train
# ----- cate1
# ------- 1-cate1.jpg
# ------- 2-cate1.jpg
# ----- cate2
# ------- 1-cate2.jpg
# ------- 2-cate2.jpg
# --- test
# ----- cate1
# ------- 1-cate1.jpg
# ------- 2-cate1.jpg
# ----- cate2
# ------- 1-cate2.jpg
# ------- 2-cate2.jpg
# --- val (可选)
# ----- cate1
# ------- 1-cate1.jpg
# ------- 2-cate1.jpg
# ----- cate2
# ------- 1-cate2.jpg
# ------- 2-cate2.jpg