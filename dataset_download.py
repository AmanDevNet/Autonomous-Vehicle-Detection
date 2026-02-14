
import os
import requests
import zipfile
import yaml
from ultralytics.utils.downloads import download
import shutil
from pathlib import Path

# Configuration
DATASET_DIR = os.path.join(os.getcwd(), "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# We will use a relevant subset of a dataset if we can, or standard COCO128 which is small and has these classes.
# However, to meet the "Autonomous Driving" vibe, we want a dataset with cars/roads.
# COCO128 contains a mix.
# A better option for a "Mini Autonomous Driving" system might be to standard COCO128 but filtered, 
# OR use a direct link to a small KITTI/BDD sample if available. 
# For reliability and speed in this demo context, we'll use COCO128 (which is standard for YOLO demos) 
# and filter/map the classes to our target list if needed, or simply download it and use the relevant classes.
# The user asked for "Cars, Bikes, Buses, Trucks, Pedestrians, Traffic lights, Stop signs"
# COCO Classes: 
# 0: person (Pedestrian)
# 1: bicycle (Bike)
# 2: car
# 3: motorcycle (Bike)
# 5: bus
# 7: truck
# 9: traffic light
# 11: stop sign

# For a more "Real-world" feel, let's try to download a specific Roboflow universe dataset or similar if we have a direct link.
# But `ultralytics` makes COCO128 trivial. Let's start with COCO128 for the base "mini" system so it definitely works without auth issues.
# We will then reshuffle it into the user's requested structure.

def download_coco128():
    print("Downloading COCO128 dataset...")
    # This downloads to current dir or ../datasets usually. We want to control it.
    # Ultralytics download function handles URLs well.
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    zip_path = "coco128.zip"
    
    if not os.path.exists(zip_path):
        download(url, dir=os.getcwd(), unzip=False)
    
    if not os.path.exists("coco128"):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
    # Move to our structure
    # coco128/images/train2017 -> dataset/images/train
    # coco128/labels/train2017 -> dataset/labels/train
    
    print("Organizing dataset...")
    base_src = "coco128"
    
    # Structure: dataset/train/images, dataset/train/labels, etc.
    # User requested:
    # dataset/
    #  ├── images/
    #  ├── labels/
    #  ├── train/ 
    #  ├── val/
    #  └── test/
    # This structure in the prompt is a bit ambiguous. Usually it's:
    # dataset/images/train, dataset/images/val
    # OR
    # dataset/train/images, dataset/train/labels
    
    # Let's go with a standard YOLOv8 structure for ease of training:
    # dataset/train/images
    # dataset/train/labels
    # dataset/val/images
    # dataset/val/labels
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATASET_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, 'labels'), exist_ok=True)

    # COCO128 only has 'train2017'. We will split it.
    src_imgs_dir = os.path.join(base_src, "images", "train2017")
    src_lbls_dir = os.path.join(base_src, "labels", "train2017")
    
    images = [f for f in os.listdir(src_imgs_dir) if f.endswith('.jpg')]
    
    # Simple split: 80% train, 10% val, 10% test
    train_split = int(0.8 * len(images))
    val_split = int(0.9 * len(images))
    
    train_imgs = images[:train_split]
    val_imgs = images[train_split:val_split]
    test_imgs = images[val_split:]
    
    def copy_files(file_list, split_name):
        for img_name in file_list:
            # Copy Image
            shutil.copy(os.path.join(src_imgs_dir, img_name), 
                        os.path.join(DATASET_DIR, split_name, 'images', img_name))
            
            # Copy Label
            lbl_name = img_name.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(src_lbls_dir, lbl_name)):
                shutil.copy(os.path.join(src_lbls_dir, lbl_name), 
                            os.path.join(DATASET_DIR, split_name, 'labels', lbl_name))
                            
    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    copy_files(test_imgs, 'test')
    
    # Create data.yaml
    data_yaml = {
        'path': DATASET_DIR,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images'),
        'nc': 80, # COCO has 80 classes. We will just map the ones we care about in inference or retraining.
        'names': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    }
    
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
        
    print("Dataset prepared successfully at", DATASET_DIR)
    
    # Pickup cleanup
    # shutil.rmtree(base_src)
    # os.remove(zip_path)

if __name__ == "__main__":
    download_coco128()
