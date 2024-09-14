from ultralytics import YOLO


if __name__ == '__main__':
    data_name = 'clothes_tags-20240731'
    train_idx = ''
    model = YOLO('models/yolov8x.pt')  # load a pretrain model
    # model = YOLO(f'runs/train/train-{data_name}{train_idx}/weights/best.pt')  # load a custom model
    results = model.train(
        # model='models/yolov8l.pt', 
        # pretrained='/data/git/yolov8/models/yolov8l-cls.pt', 
        task='detect', 
        data=f'datasets/{data_name}/data.yaml',
        epochs=300,
        batch=16,
        device=4,
        optimizer='AdamW',
        momentum=0.9,
        lr0=0.001,
        # lrf=0.01,
        lrf=0.01,
        patience=0,
        amp=False,
        degrees=100,
        project='runs/train',
        name=f'train-{data_name}')


# - clothes_tags
# --- data.yaml
# --- train
# ----- images
# ------- img1.jpg
# ------- img2.jpg
# ----- labels
# ------- img1.txt
# ------- img2.txt
# --- val
# ----- images
# ------- img1.jpg
# ------- img2.jpg
# ----- labels
# ------- img1.txt
# ------- img2.txt
# --- test（可选）
# ----- images
# ------- img1.jpg
# ------- img2.jpg
# ----- labels
# ------- img1.txt
# ------- img2.txt