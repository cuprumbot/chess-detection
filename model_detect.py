import ultralytics
import os
from ultralytics import YOLO
from PIL import Image
import cv2
from IPython.display import Video
import glob
import matplotlib.pyplot as plt
import warnings
import torch

warnings.filterwarnings('ignore')

def print_used_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory Allocated: {allocated:.2f} MB")
        print(f"GPU Memory Reserved:  {reserved:.2f} MB")

def main():
    ultralytics.checks()

    corner_model = YOLO('models/trained_model_corners.pt')
    pieces_model = YOLO('models/trained_model_pieces.pt')
    
    path = r'dataset\roboflow.jpg'

    corner = corner_model.predict(source=path, conf=0.1, save=True, project='runs', name='corner', exist_ok=True)
    pieces = pieces_model.predict(source=path, conf=0.5, save=True, project='runs', name='pieces', exist_ok=True)

    print_used_memory()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()