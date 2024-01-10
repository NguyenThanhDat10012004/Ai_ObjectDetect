
## lưu vào 1 thư mục 
```python
%cd /content/drive/MyDrive/ObjectDetection
```
## giải nén file data
```python
!unzip /content/drive/MyDrive/cars_yolo_data.zip
```
## cài đặt thư viện ultralytics để sử dụng yolov8
```python
%pip install ultralytics
import ultralytics
ultralytics.checks()
```
## import thư viện
```python
import os
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
```
## Load a model ở đây use model đã được train từ trc 
```python
yolo_yaml_path = '/content/drive/MyDrive/ObjectDetection/yolo_data/data.yml'
model = YOLO('yolov8s.yaml').load('/content/drive/MyDrive/ObjectDetection/models/yolov8/detect/train3/weights/best.pt')
```
## Train the model
```python
epochs = 25
imgsz = 640
batch_size = 8
patience = 5
lr = 0.0005
results = model.train(
    data=yolo_yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    lr0=lr,
    patience=patience,
    project='models',
    name='yolov8/detect/train'
)
```
## evaluate model arcording to mAP50, mAP50-95 > 0.5 là ổn
```python
from ultralytics import YOLO
model_path = '/content/drive/MyDrive/ObjectDetection/models/yolov8/detect/train3/weights/best.pt'
model = YOLO(model_path)
metrics = model.val(
    project='models',
    name='yolov8/detect/val'
)
```
## bounding box
```python
def visualize_bbox(
    img_path, predictions,
    conf_thres=0.8,
    font=cv2.FONT_HERSHEY_SIMPLEX
):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    for prediction in predictions:
        conf_score = prediction['confidence']
        if conf_score < conf_thres:
            continue
        bbox = prediction['box']
        xmin = int(bbox['x1'])
        ymin = int(bbox['y1'])
        xmax = int(bbox['x2'])
        ymax = int(bbox['y2'])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        text = f"{conf_score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
        cv2.rectangle(img, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), (0, 255, 0), -1)
        cv2.putText(img, text, (xmin, ymin - 5), font, 1, (0, 0, 0), 2)
    return img
```
## predict 1 album
```python
from ultralytics import YOLO
model_path = '/content/drive/MyDrive/ObjectDetection/models/yolov8/detect/train3/weights/best.pt'
test_img_dir = '/content/drive/MyDrive/ObjectDetection/yolo_data/val/images'
conf_thres=0.1
model = YOLO(model_path)
for img_name in os.listdir(test_img_dir):
    img_path = os.path.join(test_img_dir, img_name)
    # Run inference
    results = model(img_path, verbose=False)
    predictions = json.loads(results[0].tojson())
    visualized_img = visualize_bbox(img_path, predictions, conf_thres)
    plt.imshow(visualized_img)
    plt.axis('off')
    plt.show()
model = "/content/drive/MyDrive/ObjectDetection/models/yolov8/detect/train3/weights/best.pt"
test_img = "/content/test_img.jpg"
model = YOLO(model)
model.predict(test_img, save = True)
```
