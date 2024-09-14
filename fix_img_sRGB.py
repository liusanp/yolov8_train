from PIL import Image
import os

def process_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # 如果需要，可以转换图像模式
            # 修复操作，例如重新保存以去除问题元数据
            img.save(image_path)
            print(f"Processed: {image_path}")
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

def batch_process_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_image(os.path.join(root, file))

# 指定要处理的目录
batch_process_images(r"D:\Document\captcha_detection\tb_img\new_cate")
