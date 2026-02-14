
import os
import cv2
import time
from flask import Flask, render_template, request, Response, send_file, redirect, url_for
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model
model_path = os.path.join("models", "yolo_best", "weights", "best.pt")
# Fallback if training didn't run
if not os.path.exists(model_path):
    model_path = "yolov8n.pt"

print(f"Loading model from {model_path}...")
model = YOLO(model_path)

# Relevant classes for Autonomous Driving (COCO indices)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 11: stop sign
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]

def process_frame(frame):
    results = model(frame, classes=TARGET_CLASSES)
    annotated_frame = results[0].plot()
    
    # Count objects
    metrics = {
        "Car": 0, "Pedestrian": 0, "Truck": 0, "Average Conf": 0.0
    }
    
    # Simple counting stats (just a sample)
    boxes = results[0].boxes
    if boxes:
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        metrics["Average Conf"] = round(np.mean(conf), 2)
        # Map class ID to name if easy, otherwise just raw count
        
    return annotated_frame, metrics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    file.save(filepath)
    
    # Process
    img = cv2.imread(filepath)
    res_img, metrics = process_frame(img)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(output_path, res_img)
    
    return render_template('result_image.html', image_file='result.jpg', metrics=metrics)

@app.route('/image/<filename>')
def get_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    file.save(filepath)
    
    # Process Video
    output_filename = 'result_video.webm'
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # VP80 for WebM is more browser friendly than mp4v
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
    
    # Store unique IDs per class
    unique_objects = {cls_id: set() for cls_id in TARGET_CLASSES}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tracking (persist=True for video)
        results = model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False)
        annotated_frame = results[0].plot()
        
        # Count unique objects
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            
            for cls_id, track_id in zip(clss, ids):
                if int(cls_id) in unique_objects:
                    unique_objects[int(cls_id)].add(int(track_id))
        
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    
    # Prepare metrics for display
    # Map class IDs to Names
    names = model.names
    metrics = {names[cid]: len(uids) for cid, uids in unique_objects.items() if len(uids) > 0}
    
    return render_template('result_video.html', video_file=output_filename, metrics=metrics) 

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        annotated_frame, _ = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

if __name__ == "__main__":
    app.run(debug=True)
