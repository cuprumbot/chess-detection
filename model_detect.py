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
import numpy as np
from shapely.geometry import Polygon

warnings.filterwarnings('ignore')

def print_used_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory Allocated: {allocated:.2f} MB")
        print(f"GPU Memory Reserved:  {reserved:.2f} MB")

def rename_detection_image(path, output_dir, new_name):
    original_filename = os.path.basename(path)
    original_path = os.path.join(output_dir, original_filename)

    new_path = os.path.join(output_dir, new_name)

    if os.path.exists(original_path):
        os.rename(original_path, new_path)
        print(f"Renamed to {new_path}")
    else:
        print(f"File not found at {original_path}")

def order_corners(boxes):
    points = []
    sums = []
    difs = []
    ordered = [None, None, None, None]
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        x = (xyxy[0] + xyxy[2]) / 2
        y = (xyxy[1] + xyxy[3]) / 2

        points.append( [x,y] )
        sums.append( x + y )
        difs.append( x - y )

    # top-left, suma menor
    ordered[0] = points[ np.argmin(sums) ]
    # bottom-right, suma mayor
    ordered[2] = points[ np.argmax(sums) ]
    # top-right, mayor resta x - y
    ordered[1] = points[ np.argmax(difs) ]
    # bottom-left, menor resta x - y
    ordered[3] = points[ np.argmin(difs) ]

    return ordered

def draw_square(path, ordered, dir):
    # apply int()
    ordered = [tuple(map(int, pt)) for pt in ordered]
    
    img = cv2.imread(path)
    cv2.line(img, ordered[0], ordered[1], color=(0, 0, 255), thickness=3)
    cv2.line(img, ordered[1], ordered[2], color=(0, 0, 255), thickness=3)
    cv2.line(img, ordered[2], ordered[3], color=(0, 0, 255), thickness=3)
    cv2.line(img, ordered[3], ordered[0], color=(0, 0, 255), thickness=3)
    output_path = os.path.join(dir, 'outline.jpg')
    cv2.imwrite(output_path, img)

def warp_image(path, ordered, dir):
    # float32 needed
    ordered = np.array(ordered, dtype=np.float32)

    img = cv2.imread(path)
    dst_size = 1000
    dst_pts = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (dst_size, dst_size))
    output_path = os.path.join(dir, 'warped.jpg')
    cv2.imwrite(output_path, warped)

'''
0: 'Black Bishop'   1: 'Black King'     2: 'Black Knight'   3: 'Black Pawn'     4: 'Black Queen'    5: 'Black Rook'
6: 'White Bishop'   7: 'White King'     8: 'White Knight'   9: 'White Pawn'     10: 'White Queen'   11: 'White Rook'

Everything except Pawn and Knight are considered tall
Only take lower half of tall pieces
'''
names = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
tall_pieces = [0, 1, 4, 5, 6, 7, 10, 11]
board = ['........', '........', '........', '........', '........', '........', '........', '........']
def calc_candidate_positions(box, cls):
    x1, y1, x2, y2 = box

    if (cls in tall_pieces):
        ymid = (y1 + y2) / 2
        y1 = ymid

    cornersPx = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    xLeft   = int(x1 // 125)
    xRight  = int(x2 // 125)
    yTop    = int(y1 // 125)
    yBottom = int(y2 // 125)

    candidates = []
    for col in range(xLeft, xRight+1):
        for row in range(yTop, yBottom+1):
            candidates.append( [col, row] )

    return candidates, cornersPx

def calc_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union
    return iou

def calc_most_likely_square(candidates, cornersPx):
    ious = []
    for cand in candidates:
        x1 = cand[0] * 125
        x2 = x1 + 125
        y1 = cand[1] * 125
        y2 = y1 + 125
        candBox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ious.append(calc_iou(candBox, cornersPx))
    
    return candidates[ np.argmax(ious) ]




def main():
    ultralytics.checks()
    corner_model = YOLO('models/trained_model_corners.pt')
    pieces_model = YOLO('models/trained_model_pieces.pt')
    
    path = r'dataset\roboflow.jpg'

    corner = corner_model.predict(source=path, conf=0.1, save=True, project='runs', name='corner', exist_ok=False)
    pieces = pieces_model.predict(source=path, conf=0.4, save=True, project='runs', name='pieces', exist_ok=False)
    rename_detection_image(path, corner[0].save_dir, "detected.jpg")
    rename_detection_image(path, pieces[0].save_dir, "detected.jpg")

    ordered = order_corners(corner[0].boxes)    

    draw_square(path, ordered, corner[0].save_dir)
    warp_image(path, ordered, corner[0].save_dir)

    path = os.path.join(corner[0].save_dir, 'warped.jpg')
    pieces_folder = pieces[0].save_dir
    square = pieces_model.predict(
        source=path, 
        conf=0.4, 
        save=True, 
        project=os.path.dirname(pieces_folder), 
        name=os.path.basename(pieces_folder), 
        exist_ok=True
    )

    # possible squares
    points = []
    for box in square[0].boxes:
        print("---------------")

        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        conf = box.conf[0].item()         # confidence score
        cls = int(box.cls[0].item())      # class ID
        label = square[0].names[cls]      # class name

        print(f"Detected {label} at {xyxy} with confidence {conf:.2f}")
        
        candidates, cornersPx = calc_candidate_positions(xyxy, cls)
        if len(candidates) == 1:
            print(f"{label} at {candidates[0]}")
            
            y = candidates[0][0]
            x = candidates[0][1]

            newChar = names[cls]
            row = list(board[x])
            row[y] = newChar
            board[x] = ''.join(row)

        else:
            for cand in candidates:
                print(f"{label} might be at {cand}")
            most_likely = calc_most_likely_square(candidates, cornersPx)
            print(f"{label} most likely to be at {most_likely}")

            y = most_likely[0]
            x = most_likely[1]
            
            newChar = names[cls]
            row = list(board[x])
            row[y] = newChar
            board[x] = ''.join(row)

    print("---------------")
    for line in board:
        print(line)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()