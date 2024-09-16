import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt
import datetime
import torch
from ultralytics import YOLO
from generate_data.rotate_utils import rotate_image, crop_around_center, largest_rotated_rect

CLASSES = ["round-metal", "round-no-metal", "square-metal", "square-no-metal"]

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def get_mask(img):
  model = YOLO("models/seg/best.pt")
  results = model(img) 
  result = results[0]
  cv2.imwrite(f"./test_output/original_image.jpg", result.orig_img)
  masks = result.masks.data
  boxes = result.boxes.data
  clss = boxes[:, 5]
  class_idx = clss.int().cpu().numpy()[0]
  class_name = CLASSES[class_idx]
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
      x1, y1, x2, y2 = boxes[obj_index, :4].int().cpu().numpy().astype(int)
      cropped_img = cv2.bitwise_and(img, img, mask=obj_mask_resized)
      cropped_img = cropped_img[y1 - 5:y2 + 5, x1 - 5:x2 + 5]
      return cropped_img, class_name

  
def template_matching(perfect_img, isulator):
    img_rgb = cv2.cvtColor(perfect_img, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(isulator, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    W, H= template.shape[:2]

    best_match_val = -np.inf
    best_match_loc = None
    best_match_angle = 0
    blurred_image_1 = cv2.blur(img,(3, 3))
    blurred_image_2 = cv2.blur(template,(3, 3))
    edges1 = cv2.Canny(blurred_image_1, 50, 100, 15)
    edges2 = cv2.Canny(blurred_image_2, 50, 100, 15)
    cv2.imwrite(f"./tm/img.jpg", edges1)
    cv2.imwrite(f"./tm/template.jpg", edges2)

    for angle in frange(0.0, 360.0, 0.2):
        try:
            rotated_template = rotate_image(template, angle)
            image_rotated_cropped = crop_around_center(
                    rotated_template,
                    *largest_rotated_rect(
                        W,
                        H,
                        math.radians(angle)
                    ))

            w, h = image_rotated_cropped.shape[:2]
            # Draw the best match rectangle on the image
            blurred_image_1 = cv2.blur(image_rotated_cropped,(5,5))
            blurred_image_2 = cv2.blur(img,(5,5))
            edges1 = cv2.Canny(blurred_image_1,  10, 150, 15)
            edges2 = cv2.Canny(blurred_image_2,  10, 150, 15)
            result = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            img1 = img.copy()
            blurred_image_2 = cv2.medianBlur(img1,7)
            edges2 = cv2.Canny(blurred_image_2, 0, 150)
            # cropped_image_sample = edges2[max_loc[1]:max_loc[1] + w, max_loc[0] :max_loc[0] + h]
            cv2.imwrite(f"./tm/edges1.jpg", edges1)
            cv2.imwrite(f"./tm/edges2.jpg", edges2)
            cv2.imwrite(f"./tm/rotate.jpg", image_rotated_cropped)
            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_match_angle = angle
        except Exception:
            continue       
    bottom_right = (best_match_loc[0] + w, best_match_loc[1] + h)
    cv2.imwrite(f"./tm/img.jpg", img)
    rotated_template = rotate_image(template_rgb, best_match_angle)
    cv2.imwrite(f"./tm/rotated_template.jpg", rotated_template)
    image_rotated_cropped = crop_around_center(
                rotated_template,
                *largest_rotated_rect(
                    W,
                    H,
                    math.radians(best_match_angle)
                ))
    w, h = image_rotated_cropped.shape[:2]
    cropped_image_sample = img_rgb[best_match_loc[1]:best_match_loc[1] + w, best_match_loc[0] :best_match_loc[0] + h]
   
    print(f'Rotation angle for best match: {best_match_angle} degrees')

    blurred_image_1 = cv2.blur(cropped_image_sample,(3, 3))
    blurred_image_2 = cv2.blur(image_rotated_cropped,(3, 3))
    edges1 = cv2.Canny(cropped_image_sample, 10, 100, 3)
    edges2 = cv2.Canny(image_rotated_cropped, 10, 100, 3)
    cv2.imwrite(f"./tm/edges1.jpg", edges1)
    cv2.imwrite(f"./tm/edges2.jpg", edges2)
    # Convert the edge images to binary sequences
    edges1_binary = np.ravel(edges1) // 255
    edges2_binary = np.ravel(edges2) // 255

    # # Compute the Hamming distance between the two binary sequences
    hamming_distance = np.count_nonzero(edges1_binary != edges2_binary)

    print("Hamming distance:", hamming_distance)
    print(cropped_image_sample.shape, template.shape)
    return hamming_distance  
        


def main():
    start = datetime.datetime.now()
    img= cv2.imread('test/no_fon.jpg')
    isulator, class_name = get_mask(img)
    if class_name == "square-metal":
        print('форма изолятора: квадратная\nсторона: с металлизацией')
        perfect_img= cv2.imread('perfect/square-metal.jpg')
    if class_name == "square-no-metal":
        print('форма изолятора: квадратная\nсторона: без металлизации')
        perfect_img= cv2.imread('perfect/square-no-metal.jpg')
    if class_name == "round-metal":
        print('форма изолятора: круглая\nсторона: с металлизацией')
        perfect_img= cv2.imread('perfect/round-metal.jpg')
    if class_name == "round-no-metal":
        print('форма изолятора: круглая\nсторона: без металлизации')
        perfect_img= cv2.imread('perfect/round-no-metal.jpg')
    perfect_img,_ = get_mask(perfect_img) 
    max_height = max(isulator.shape[0], perfect_img.shape[0])
    max_width = max(isulator.shape[1], perfect_img.shape[1])

    # Создать новые изображения с одинаковыми размерами
    isulator_new = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    perfect_img_new = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Центрировать изображения на новом фоне
    h1, w1 = isulator.shape[:2]
    h2, w2 = perfect_img.shape[:2]
    x1 = (max_width - w1) // 2
    y1 = (max_height - h1) // 2
    x2 = (max_width - w2) // 2
    y2 = (max_height - h2) // 2

    isulator_new[y1:y1+h1, x1:x1+w1] = isulator
    perfect_img_new[y2:y2+h2, x2:x2+w2] = perfect_img

    top, bottom, left, right = 100, 100, 100, 100

    # Расширение изображения
    perfect_img_new = cv2.copyMakeBorder(perfect_img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    isulator_new = cv2.copyMakeBorder(isulator_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite(f"./tm/perf.jpg", perfect_img_new)  
    cv2.imwrite(f"./tm/isulator.jpg", isulator_new)  
    hamming_distance = template_matching(perfect_img_new, isulator_new) 
    if class_name == "square-metal":
        threshold = 4000
    if class_name == "square-no-metal":
        threshold = 2500
    if class_name == "round-metal":
        threshold = 1300
    if class_name == "round-no-metal":
        threshold = 2000
    if hamming_distance > threshold:
        print('дефект')
    else:
        print('дефекта не обнаружено')
    
    finish = datetime.datetime.now()   
    print('Время работы: ' + str(finish - start))  
if __name__ == "__main__":
    main()   