import glob
from ultralytics import YOLO
import datetime
import cv2

start = datetime.datetime.now()
model = YOLO("models/seg/best.pt") 
images = glob.glob("./test/*.jpg")
results = model(images)
i = 0
print(len(images))
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    
    # Отрисовка и сохранение маски
    for j, mask in enumerate(masks.data):
        mask_image = (mask.cpu().numpy() * 255).astype('uint8')
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"./result/result_{i}_mask_{j}.jpg", mask_image)
    
    # Сохранение изображения с boundary box, метками классов и масками
    result.save(filename=f"./result/result_{i}.jpg")  
    i += 1

finish = datetime.datetime.now()       
print('Время работы: ' + str(finish - start))  
