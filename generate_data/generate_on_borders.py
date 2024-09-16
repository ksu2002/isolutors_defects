'''
Наложение металлизации на края и осколков(фона).
'''
from PIL import Image, ImageDraw
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt 
from generate_data.DefectCreator import add_defects
import torch
import random
import os
import numpy as np 
CLASSES = ["round-metal", "round-no-metal", "square-metal", "square-no-metal"]

def crop_and_overlay(image, mask, new_image):
    image = np.array(image)
    new_image = np.array(new_image)
    # Проверяем, что размеры всех изображений совпадают

    # Создаем копию исходного изображения
    result = image.copy()

    # Используем Numpy-совместимую индексацию
    result[mask == 1] = new_image[mask == 1]
    img = Image.fromarray((result.astype(np.uint8)).reshape((result.shape[0], result.shape[1], 3)), mode='RGB')

    return img



def get_mask_aug(img_path):
    img= cv2.imread(img_path)
    model = YOLO("./models/seg/best.pt")
    results = model(img)
    result = results[0]
    masks = result.masks.data
    boxes = result.boxes.data
    clss = boxes[:, 5]
    class_idx = clss.int().cpu().numpy()[0]
    class_name = CLASSES[class_idx]
    print(class_name)
    obj_indices = torch.where(clss != -1)
    obj_masks = masks[obj_indices]
    obj_mask = torch.any(obj_masks, dim=0).int() * 255
    obj_mask = torch.any(obj_masks, dim=0).int()
    for i, obj_index in enumerate(obj_indices[0].cpu().numpy()):
        obj_masks = masks[torch.tensor([obj_index])]
        obj_mask = torch.any(obj_masks, dim=0).int()
        obj_mask = obj_mask.cpu().numpy()
        obj_mask_uint8 = obj_mask.astype(np.uint8)
        obj_mask_resized = cv2.resize(obj_mask_uint8, (img.shape[1], img.shape[0]))
        plt.imshow(obj_mask_resized)
        plt.show()
        cropped_img = cv2.bitwise_and(img, img, mask=obj_mask_resized)
        image_orig = Image.open(img_path)
        image1, x1, y1, x2, y2 = add_defects(image_orig, obj_mask_resized)
        image_orig = Image.open(img_path)
        image = crop_and_overlay(image_orig, obj_mask_resized,image1)
        image.save("./defect.jpg")
        print(x1, y1, x2, y2)
        
        # draw = ImageDraw.Draw(image)
        # draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=5)
        
        x_center = float((x1 + ((x2-x1)/2))/image.size[1])
        y_center = float((y1 + ((y2-y1)/2))/image.size[0])
        width = float(x2-x1)/image.size[1]
        height = float(y2-y1)/image.size[0]
        defect_class = 1
        annotation = f'{defect_class} {x_center} {y_center} {width} {height}'
        return image, annotation

def main():
    folderpath = 'generated_no_defect'

    # files = [f'perf_round_metal_{i+1}.jpg' for i in range(516)]
    # files = [f'perf_round_no_metal_{i+1}.jpg' for i in range(524)]
    # files = [f'perf_square_no_metal_{i+1}.jpg' for i in range(500)]
    files = [f'perf_square_metal_{i+1}.jpg' for i in range(572)]


    # img_path = f"./perfect/square-metal.jpg"
    for i in range(500, 650):
        try:
            randomfile = random.choice(files)
            img_path = os.path.join(folderpath, randomfile)
            folder = 'train'
            if (i % 10 == 0):
                folder = 'valid'
            isulator, annotation = get_mask_aug(img_path)
            if isulator is None:
                continue
            # Сохранение изображения
            img_filename = f"dataset/{folder}/images/square_metal_{i}.jpg"
            isulator.save(img_filename)
            print(img_filename)
            # Сохранение текстового файла
            txt_filename = f"dataset/{folder}/labels/square_metal_{i}.txt"
            with open(txt_filename, 'w') as f:
                f.writelines(annotation)
        except Exception:
            continue        
if __name__ == "__main__":
    main()        