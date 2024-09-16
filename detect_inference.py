import glob
from ultralytics import YOLO
import datetime
start = datetime.datetime.now()
# Load a model
model = YOLO("models/detect/with classification/best.pt")  # pretrained YOLOv8n model

# Get all jpg images in the test folder
images = glob.glob("./test/*.jpg")
# Run batched inference on the list of images
results = model(images, conf=0.5)  # return a list of Results objects
i = 0
# image = 'test/IMG_20240323_1629136.jpg'
# result = model(image)
# finish = datetime.datetime.now()   
# result.save(filename=f"./result.jpg")
# print('Время работы: ' + str(finish - start)) 
print(len(images))
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.save(filename=f"./result/result_{i}.jpg")
    i += 1
finish = datetime.datetime.now()       
print('Время работы: ' + str(finish - start))     
