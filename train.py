
import argparse
from ultralytics import YOLO
import torch
import torchvision
import os

def train_yolo(epochs=10):
    print("Starting YOLOv8 Training...")
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # We point to the data.yaml created by dataset_download.py
    data_path = os.path.join(os.getcwd(), "dataset", "data.yaml")
    
    results = model.train(data=data_path, epochs=epochs, imgsz=640, project="models", name="yolo_best", exist_ok=True)
    
    # Export the model
    success = model.export(format="torchscript")
    print("YOLO Training complete.")
    return model

def train_faster_rcnn(epochs=1):
    print("Starting Faster R-CNN Training (Demo)...")
    # This involves setting up a rigorous PyTorch training loop. 
    # For a "mini" system, we might just load a pretrained one and fine-tune slightly or just save it.
    # To keep this script runnable without massive complexity, we will save a pretrained Faster R-CNN.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Save parameters
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fasterrcnn.pt")
    print("Faster R-CNN model saved (Pretrained).")

def train_ssd(epochs=1):
    print("Starting SSD Training (Demo)...")
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ssd.pt")
    print("SSD model saved (Pretrained).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for YOLO")
    args = parser.parse_args()
    
    # Train Primary
    train_yolo(epochs=args.epochs)
    
    # Train Secondary (Demo/Pretrained for comparison logic)
    train_faster_rcnn()
    train_ssd()
