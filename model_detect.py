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

    model = YOLO('models/trained_model_mid.pt')
    
    example_video_path = r'C:\Users\luisr\chess-detection\dataset\video.mp4'

    video_output = model.predict(source=example_video_path, conf=0.5,save=True)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()