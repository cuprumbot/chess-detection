import ultralytics
import os
from ultralytics import YOLO
from PIL import Image
import cv2
from IPython.display import Video
import glob
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def main():
    ultralytics.checks()

    model = YOLO('yolov8m.pt')

    root_dir = r'C:\Users\luisr\chess-detection\dataset'
    yaml_path = os.path.join(root_dir, 'data.yaml.yaml')
    train_path = os.path.join(root_dir, 'train', 'images')
    valid_path = os.path.join(root_dir, 'valid', 'images')

    results = model.train(
        data=yaml_path,
        epochs=20,
        batch=8,
        lr0=0.0001,
        lrf=0.1,
        imgsz=512,
        plots=True
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()